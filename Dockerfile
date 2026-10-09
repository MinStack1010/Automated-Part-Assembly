# syntax=docker/dockerfile:1

########################################################################
# Stage 1: builder - build redmax_py (CMake + pybind11) và cài deps
########################################################################
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libgl1-mesa-dri \
        libglu1-mesa-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-venv \
        xauth \
        xorg-dev \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    LIBGL_ALWAYS_SOFTWARE=1 \
    DISPLAY=:99 \
    XVFB_AUTOSTART=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_STORAGE_DIR=/var/lib/assemble-them-all \
    ASSEMBLY_ASSETS_DIR=/app/assets

# Entry point tự start Xvfb -> chạy headless được ngay cả khi là PID 1
# (xvfb-run thì treo waiting SIGUSR1 khi là PID 1).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /app

# Chỉ copy thư mục simulation để context nhỏ và layer cache tốt.
# QUAN TRỌNG: đường dẫn /app/simulation được bake vào binary
# (GRAPHICS_CODEBASE_SOURCE_DIR) -> runtime image phải giữ đúng path này.
# Đặt bước build C++ (~12 phút) TRƯỚC requirements để đổi requirements.txt
# không phải compile lại từ đầu.
COPY simulation ./simulation
RUN pip install --no-cache-dir ./simulation \
    && rm -rf /app/simulation/build /app/simulation/dist /app/simulation/redmax_py.egg-info

COPY requirements.txt .

# ghalton (baseline planner) chỉ có sdist và hardcode flag clang '-stdlib=libc++'
# -> dùng shim compiler để loại flag đó khi build với gcc.
RUN mkdir -p /opt/cc-shim \
    && printf '%s\n' \
       '#!/bin/bash' \
       'args=()' \
       'for a in "$@"; do [[ "$a" == "-stdlib=libc++" ]] && continue; args+=("$a"); done' \
       'exec /usr/bin/gcc "${args[@]}"' > /opt/cc-shim/gcc \
    && printf '%s\n' \
       '#!/bin/bash' \
       'args=()' \
       'for a in "$@"; do [[ "$a" == "-stdlib=libc++" ]] && continue; args+=("$a"); done' \
       'exec /usr/bin/g++ "${args[@]}"' > /opt/cc-shim/g++ \
    && chmod +x /opt/cc-shim/gcc /opt/cc-shim/g++ \
    && CC=/opt/cc-shim/gcc CXX=/opt/cc-shim/g++ pip install --no-cache-dir ghalton==0.6.2

RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import redmax_py; print('redmax_py imported OK')"

########################################################################
# Stage 2: runtime - chỉ có python + libs chạy (không toolchain build)
########################################################################
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglu1-mesa \
        libgomp1 \
        libice6 \
        libsm6 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxxf86vm1 \
        python3 \
        python3-tk \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    LIBGL_ALWAYS_SOFTWARE=1 \
    DISPLAY=:99 \
    XVFB_AUTOSTART=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_STORAGE_DIR=/var/lib/assemble-them-all \
    ASSEMBLY_ASSETS_DIR=/app/assets

# Entry point tự start Xvfb -> chạy được headless, không cần xvfb-run (và
# xvfb-run thì treo khi là PID 1 của container).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Nguồn code: path phải giống hệt lúc build (xem chú thích ở builder).
COPY . /app

RUN mkdir -p "$API_STORAGE_DIR"

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "server.py"]
