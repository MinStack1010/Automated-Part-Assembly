#!/bin/sh
# Entry point: tự khởi động Xvfb (X server ảo) để chạy headless trên Docker.
# - Mặc định: DISPLAY=:99, Xvfb được start trước khi chạy lệnh.
# - Khi đã có X server thật (service `gui`): đặt XVFB_AUTOSTART=0.
set -e

start_xvfb() {
    display_num=$(printf '%s' "${DISPLAY:-:99}" | sed 's/^://; s/\..*$//')
    sock="/tmp/.X11-unix/X${display_num}"

    if [ -S "$sock" ]; then
        return 0
    fi

    Xvfb "$DISPLAY" -screen 0 "${XVFB_SCREEN:-1280x1024x24}" -nolisten tcp \
        >/tmp/xvfb.log 2>&1 &
    xvfb_pid=$!

    i=0
    while [ "$i" -lt 50 ]; do
        if [ -S "$sock" ]; then
            return 0
        fi
        if ! kill -0 "$xvfb_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
        i=$((i + 1))
    done

    echo "entrypoint: khong khoi duoc Xvfb tren $DISPLAY" >&2
    cat /tmp/xvfb.log >&2 || true
    return 1
}

if [ "${XVFB_AUTOSTART:-1}" = "1" ]; then
    start_xvfb
fi

exec "$@"
