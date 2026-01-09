# HSLU CaptureStudio: A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats

This is the **official code repository** for the paper:

> **A Fast Volumetric Capture and Reconstruction Pipeline for Dynamic Point Clouds and Gaussian Splats**  
> Athanasios Charisoudis, Simone Croci, Lam Kit Yung, Pascal Frossard, Aljosa Smolic  
> *European Conference on Visual Media Production (CVMP ’25)*  
> DOI: [10.1145/3756863.3769713](https://doi.org/10.1145/3756863.3769713)

Project page: https://irc-hslu.github.io/capturestudio  
Paper: https://doi.org/10.1145/3756863.3769713  

---

## Code Release Status

The code for the HSLU CaptureStudio pipeline (capture, reconstruction, and export of dynamic point clouds and Gaussian splats) is currently being:

-[x] cleaned up and modularized
-[x] documented
-[x] prepared for a public release
-[x] initial version out (v1)
-[ ] incorporate v2 improvements (in progress)
-[ ] complete requirements.txt
-[ ] scripts for reconstruction and export

---

## Early Access / Urgent Requests

If you urgently need access to parts of the code, or have specific research / industry use cases, feel free to reach out:

**Athanasios Charisoudis**  
Lucerne University of Applied Sciences and Arts (HSLU)  
📧 athanasios.charisoudis@hslu.ch

I’m happy to discuss:

- early access to parts of the pipeline (where possible)
- collaborations
- reproducibility questions
- integration into your own projects

---

## Repo Structure

The repository is organized as follows:
- `docs/`: Project website (based on aifolio template)
- `recon-viewer/`: Interactive viewer for reconstructed dynamic point clouds and Gaussian splats
- `src/`: Main source code for the HSLU CaptureStudio pipeline

---

## How to Run the Celery-based Processing and Reconstruction Pipeline

1. Install celery system-wide:
   ```bash
   sudo apt update
   sudo apt install -y celery
   ```
2. Install the Python dependencies:
   ```bash
   python -m pip install celery redis
   ```

   3. Open 3 terminals and run the following commands in each (after cding to the `src` directory):
        Start 12 cpu workers:
      ```bash
       celery -A tasks worker --loglevel=INFO --concurrency=12 --max-tasks-per-child=1 -Q cpu --hostname=cpu@%h
       ```
        Start 2 gpu workers:
       ```bash
       celery -A tasks worker --loglevel=INFO --concurrency=2 --max-tasks-per-child=1 -Q gpu --hostname=gpu@%h
       ```
        Start flower monitoring tool:
       ```bash 
       celery -A tasks flower --port=5555
       ```

4. Run the submission script to start the tasks:
   ```bash
   python src/_misc/submission_scripts/apr_may_2025.py
   ```
   Edit the `src/_misc/submission_scripts/apr_may_2025.py` by providing the performances that you want to run the tasks on.
    
    See also other scripts in src/_misc/submission_scripts/

5. Open your browser and go to `http://localhost:5555` to access the Flower monitoring tool.


6. Handling failed tasks:
    If any tasks fail, you can retry them by first restarting celery workers and run the submission script (successfully finished tasks are not redone). To do that, first clear the queues by running in src directory:
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
