# Docker-Related Env Config

## Docker Installation (Include NVIDIA Docker)

1. **更新 apt 包索引并安装依赖：**

   ```bash
   sudo apt-get update
   sudo apt-get install \
       ca-certificates \
       curl \
       gnupg \
       lsb-release
   ```

2. **添加 Docker 的官方 GPG 密钥：**

   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

3. **设置 Docker 的稳定版本存储库：**

   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

4. **安装 Docker 引擎：**

   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

5. **启动 Docker 并设置开机启动：**

   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

6. **验证 Docker 安装：**

   ```bash
   sudo docker run hello-world
   ```

> 如果我们需要配置深度学习相关的 docker 环境，我们还需要额外配置 nvidia docker（用来使用 CUDA）
>
> ```bash
> distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
> && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
> && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
> 
> sudo apt-get update
> sudo apt-get install -y nvidia-docker2
> sudo systemctl restart docker
> ```

