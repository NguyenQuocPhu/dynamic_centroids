# Dev container for dynamic_centroids: CUDA 13 + a passwordless-sudo user.
# Mounts the 3 branch worktrees (main / beam-search / mcts) side by side —
# see docker-compose.yml. No agent/LLM CLI is pre-installed; install whatever
# you want to use (claude, aider, ...) once you're inside the container.

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        git \
        curl \
        wget \
        vim \
        build-essential \
        python3 \
        python3-venv \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with passwordless sudo. UID/GID default to 1000 but are
# overridable at build time to match the host user (see docker-compose.yml),
# so files written into the bind-mounted worktrees keep sane ownership.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Convenience symlinks. The real mounts land at their original host paths
# (see docker-compose.yml comment) because git worktree metadata embeds
# absolute paths; these just give short, memorable names.
RUN mkdir -p /workspace \
    && ln -s /home/phunq/5080_cuda13/Research/dynamic_centroids /workspace/main \
    && ln -s /tmp/wt-beam3 /workspace/beam \
    && ln -s /tmp/wt-mcts3 /workspace/mcts \
    && chown -h ${USERNAME}:${USERNAME} /workspace/main /workspace/beam /workspace/mcts

USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]
