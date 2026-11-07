#!/usr/bin/env bash
# NVMe-oF lab for two Linux VMs. It serves a RAM disk, so the disk itself isn't the bottleneck and
# what's left is the cost of the storage protocol and the network path.
#
#   Target VM:    sudo scripts/nvmeof_lab.sh target tcp
#   Initiator VM: sudo scripts/nvmeof_lab.sh bench tcp TARGET_IP OUT_DIR
#
# Use rdma in place of tcp for Soft-RoCE. Run `local` on the target for the no-network baseline.
# Soft-RoCE is RDMA emulated in software, so its numbers say nothing about real RDMA hardware.
set -euo pipefail

MODE=${1:?target, bench, local or clean}
TRANSPORT=${2:-tcp}
TARGET_IP=${3:-}
OUT_DIR=${4:-/tmp/nvmeof_results}
NQN=nqn.2026-11.io.anton:ram
PORT_ID=1
SVC_PORT=4420
RAM_MB=${RAM_MB:-512}
CFG=/sys/kernel/config/nvmet

ensure_rxe() {
    # Soft-RoCE runs over an ordinary network interface.
    modprobe rdma_rxe
    if ! rdma link show 2>/dev/null | grep -q rxe0; then
        # Use the interface shared with the other VM, which comes last in `ip` output here.
        local netdev
        netdev=$(ip -o link show lima0 >/dev/null 2>&1 && echo lima0 || ip -4 -br addr | awk '$1 != "lo" {print $1}' | tail -1 | cut -d@ -f1)
        rdma link add rxe0 type rxe netdev "$netdev"
    fi
}

clean_target() {
    rm -f "$CFG/ports/$PORT_ID/subsystems/$NQN" 2>/dev/null || true
    rmdir "$CFG/ports/$PORT_ID" 2>/dev/null || true
    if [ -d "$CFG/subsystems/$NQN" ]; then
        echo 0 >"$CFG/subsystems/$NQN/namespaces/1/enable" 2>/dev/null || true
        rmdir "$CFG/subsystems/$NQN/namespaces/1" 2>/dev/null || true
        rmdir "$CFG/subsystems/$NQN" 2>/dev/null || true
    fi
}

run_fio() { # name device
    local name=$1 device=$2
    mkdir -p "$OUT_DIR"
    # 32 MiB matches one LMCache chunk for this model: 256 tokens x 128 KiB per token.
    fio --name="$name" --filename="$device" --rw=read --bs=32M --iodepth=4 --direct=1 \
        --ioengine=libaio --runtime=20 --time_based --group_reporting --output-format=json \
        >"$OUT_DIR/${name}_chunk_32m.json"
    # 4 KiB at depth 1 shows the round-trip latency of the path, with little data moved.
    fio --name="$name" --filename="$device" --rw=randread --bs=4k --iodepth=1 --direct=1 \
        --ioengine=libaio --runtime=20 --time_based --group_reporting --output-format=json \
        >"$OUT_DIR/${name}_lat_4k.json"
    # One whole restore of a median agent session: 11,788 tokens x 128 KiB, in LMCache chunks.
    fio --name="$name" --filename="$device" --rw=read --bs=32M --iodepth=4 --direct=1 \
        --ioengine=libaio --size=1472M --loops=5 --group_reporting --output-format=json \
        >"$OUT_DIR/${name}_restore_session.json"
}

case "$MODE" in
target)
    modprobe nvmet
    modprobe "nvmet-$TRANSPORT"
    modprobe brd rd_nr=1 rd_size=$((RAM_MB * 1024)) max_part=0
    [ "$TRANSPORT" = rdma ] && ensure_rxe
    clean_target
    IP=$(ip -4 addr show lima0 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1 || true)
    [ -n "$IP" ] || IP=$(ip -4 -br addr | awk '$1 != "lo" {print $3}' | tail -1 | cut -d/ -f1)
    mkdir -p "$CFG/subsystems/$NQN/namespaces/1"
    echo 1 >"$CFG/subsystems/$NQN/attr_allow_any_host"
    echo /dev/ram0 >"$CFG/subsystems/$NQN/namespaces/1/device_path"
    echo 1 >"$CFG/subsystems/$NQN/namespaces/1/enable"
    mkdir -p "$CFG/ports/$PORT_ID"
    echo "$IP" >"$CFG/ports/$PORT_ID/addr_traddr"
    echo "$TRANSPORT" >"$CFG/ports/$PORT_ID/addr_trtype"
    echo "$SVC_PORT" >"$CFG/ports/$PORT_ID/addr_trsvcid"
    echo ipv4 >"$CFG/ports/$PORT_ID/addr_adrfam"
    ln -s "$CFG/subsystems/$NQN" "$CFG/ports/$PORT_ID/subsystems/$NQN"
    echo "serving $NQN over $TRANSPORT at $IP:$SVC_PORT"
    ;;
local)
    modprobe brd rd_nr=1 rd_size=$((RAM_MB * 1024)) max_part=0
    run_fio local_ram /dev/ram0
    ;;
bench)
    modprobe "nvme-$TRANSPORT"
    [ "$TRANSPORT" = rdma ] && ensure_rxe
    nvme disconnect -n "$NQN" >/dev/null 2>&1 || true
    nvme connect -t "$TRANSPORT" -a "$TARGET_IP" -s "$SVC_PORT" -n "$NQN"
    sleep 2
    # Find it by name: on cloud VMs, the boot disk is also an NVMe device.
    DEVICE=
    for controller in /sys/class/nvme/nvme*; do
        if [ "$(cat "$controller/subsysnqn")" = "$NQN" ]; then
            DEVICE=/dev/$(basename "$controller")n1
        fi
    done
    [ -n "$DEVICE" ] || { echo "no device for $NQN" >&2; exit 1; }
    echo "testing $DEVICE"
    run_fio "nvmeof_$TRANSPORT" "$DEVICE"
    nvme disconnect -n "$NQN"
    ;;
clean)
    clean_target
    ;;
*)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
