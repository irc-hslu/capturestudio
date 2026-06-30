# HSLU CaptureStudio: A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats

This is the **official code repository** for the paper:

> **A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats**  
> Athanasios Charisoudis, Simone Croci, Lam Kit Yung, Pascal Frossard, Aljosa Smolic  
> *European Conference on Visual Media Production (CVMP '25)*  
> DOI: [10.1145/3756863.3769713](https://doi.org/10.1145/3756863.3769713)

Project page: https://irc-hslu.github.io/capturestudio  
Paper: https://doi.org/10.1145/3756863.3769713  

---

## Light Variant for Multi-RGBD ORBBEC Capture

If you only want to capture using multiple RGBD ORBBEC sensors, please see the light variant, which is easier to install and run:

https://github.com/irc-hslu/capturestudio-light

---

## Code Release Status

The code for the HSLU CaptureStudio pipeline (capture, reconstruction, and export of dynamic point clouds and Gaussian splats) is currently being:

- [x] cleaned up and modularized
- [x] documented
- [x] prepared for a public release
- [x] initial version out (v1)
- [x] incorporated v2 improvements
- [x] complete `requirements.txt` > `pyproject.toml` (using **uv**)
- [ ] scripts for reconstruction and export

---

## GPSGaussian Model Checkpoints

The checkpoints for the **GPSGaussian** model variants used in the paper are hosted on Hugging Face:

**Hugging Face:** https://huggingface.co/irc-hslu/GPSGaussian

Please refer to the model card and repository contents there for the available variants and checkpoint files.

---

## Repo Structure

The repository is organized as follows:

- `docs/`: Project website (based on aifolio template)
- `recon-viewer/`: Interactive viewer for reconstructed dynamic point clouds and Gaussian splats
- `src/`: Main source code for the HSLU CaptureStudio pipeline

---

## How to Run the Celery-based Processing and Reconstruction Pipeline

0. Install celery dependencies:

   i. `redis`
```bash
sudo apt-get install lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis
```

   ii. `rabbitmq`
```bash
sudo apt-get install curl gnupg apt-transport-https -y

## Team RabbitMQ's signing key
curl -1sLf "https://keys.openpgp.org/vks/v1/by-fingerprint/0A9AF2115F4687BD29803A206B73A36E6026DFCA" | sudo gpg --dearmor | sudo tee /usr/share/keyrings/com.rabbitmq.team.gpg > /dev/null
   
## Add apt repositories maintained by Team RabbitMQ
sudo tee /etc/apt/sources.list.d/rabbitmq.list <<EOF
## Modern Erlang/OTP releases
##
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb1.rabbitmq.com/rabbitmq-erlang/ubuntu/noble noble main
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb2.rabbitmq.com/rabbitmq-erlang/ubuntu/noble noble main
   
## Latest RabbitMQ releases
##
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb1.rabbitmq.com/rabbitmq-server/ubuntu/noble noble main
deb [arch=amd64 signed-by=/usr/share/keyrings/com.rabbitmq.team.gpg] https://deb2.rabbitmq.com/rabbitmq-server/ubuntu/noble noble main
EOF
      
## Update package indices
sudo apt-get update -y
   
## Install Erlang packages
sudo apt-get install -y erlang-base \
                        erlang-asn1 erlang-crypto erlang-eldap erlang-ftp erlang-inets \
                        erlang-mnesia erlang-os-mon erlang-parsetools erlang-public-key \
                        erlang-runtime-tools erlang-snmp erlang-ssl \
                        erlang-syntax-tools erlang-tftp erlang-tools erlang-xmerl
   
## Install rabbitmq-server and its dependencies
sudo apt-get install rabbitmq-server -y --fix-missing
```

1. Install Celery system-wide:
   ```bash
   sudo apt update
   sudo apt install -y celery
   ```

2. Install the Python dependencies:
   ```bash
   python -m pip install celery redis
   ```

3. Open 3 terminals and run the following commands in each (after changing to the `src` directory):

   Start 12 CPU workers:
   ```bash
   celery -A tasks worker --loglevel=INFO --concurrency=12 --max-tasks-per-child=1 -Q cpu --hostname=cpu@%h
   ```

   Start 2 GPU workers:
   ```bash
   celery -A tasks worker --loglevel=INFO --concurrency=2 --max-tasks-per-child=1 -Q gpu --hostname=gpu@%h
   ```

   Start Flower monitoring tool:
   ```bash
   celery -A tasks flower --port=5555
   ```

4. Run the submission script to start the tasks:
   ```bash
   python src/_misc/submission_scripts/apr_may_2025.py
   ```

   Edit `src/_misc/submission_scripts/apr_may_2025.py` by providing the performances that you want to run the tasks on.

   See also other scripts in `src/_misc/submission_scripts/`.

5. Open your browser and go to `http://localhost:5555` to access the Flower monitoring tool.

6. Handling failed tasks:

   If any tasks fail, you can retry them by first restarting Celery workers and running the submission script again (successfully finished tasks are not redone). To do that, first clear the queues by running in the `src` directory:
   ```bash
   sudo rabbitmqctl purge_queue cpu && sudo rabbitmqctl purge_queue gpu && redis-cli flushdb && celery -A tasks control shutdown
   ```

---

## Work under active development

Please note that this project is under active development, and the code may change frequently. If you encounter any issues or have suggestions, feel free to open an issue on the project's GitHub repository.

---

## How to Cite

If you use this work in your research, please cite:

```bibtex
@inproceedings{10.1145/3756863.3769713,
    author = {Charisoudis, Athanasios and Croci, Simone and Lam, Kit Yung and Frossard, Pascal and Smolic, Aljosa},
    title = {A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats},
    year = {2025},
    isbn = {9798400721175},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3756863.3769713},
    doi = {10.1145/3756863.3769713},
    booktitle = {Proceedings of the 22nd ACM SIGGRAPH European Conference on Visual Media Production},
    articleno = {9},
    numpages = {11},
    keywords = {Volumetric video capture, point clouds, Gaussian splats, dynamic reconstruction},
    series = {CVMP '25}
}
```

---

## Contact

**Athanasios Charisoudis**  
Immersive Realities Center, Hochschule Luzern  
athanasios.charisoudis@hslu.ch