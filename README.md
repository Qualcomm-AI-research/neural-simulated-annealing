# Neural Simulated Annealing
This repository will contain the implementation for the paper

**Alvaro H.C. Correia<sup>\*1</sup>, Daniel E. Worrall<sup>\*2</sup>, Roberto Bondesan<sup>2</sup> "Neural Simulated Annealing".** [[ArXiv]](https://arxiv.org/abs/2203.02201)

*Equal contribution

<sup>1</sup> Eindhoven University of Technology, Eindhoven, The Netherlands (Work done during internship at Qualcomm AI Research).

<sup>2</sup> Qualcomm AI Research, Qualcomm Technologies Netherlands B.V. (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.).

 ## Reference
If you find our work useful, please cite
```
@inproceedings{correia2023neural,
  title={Neural simulated annealing},
  author={Correia, Alvaro HC and Worrall, Daniel E and Bondesan, Roberto},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4946--4962},
  year={2023},
  organization={PMLR}
}
```

## How to install

Make sure to have Python ≥3.10 (tested with Python 3.10.11) and 
ensure the latest version of `pip` (tested with 22.3.1):
```bash
pip install --upgrade --no-deps pip
```

Next, install PyTorch 1.13.0 with the appropriate CUDA version (tested with CUDA 11.7):
```bash
python -m pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```

To run the code, the project root directory needs to be added to your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

## Running experiments
### Training
The main run file to reproduce all experiments is `main.py`. We use [Hydra](https://hydra.cc/) to configure experiments, so you can retrain our Neural SA models as follows
```bash
python scripts/main.py +experiment=<config_file>
```
where <config_file> is a yaml file defining the experiment configuration. The experiments in the paper are configured via the config files in the `scripts/conf/experiment` folder, which are named as `<problem>_<method>.yaml`. For instance, to train a Knapsack model using PPO with the configuration used in the paper you should run
```bash
python scripts/main.py +experiment=knapsack_ppo
```

To experiment with different configurations, you can either create a new yaml file and use it on the command line as above, or you can change specific variables with [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). As an example, if you want to keep the same configuration of the Knapsack experiments with PPO but train with problems of size 100 and for 500 steps, you can do so as follows

```bash
python scripts/main.py +experiment=knapsack_ppo ++problem_dim=100 ++sa.outer_steps=500
```

See neuralsa/config.py for an overview of the different configuration variables. Note that for parameters in the SAConfig class you need to prepend its name with "sa" (as in the example above), and for the TrainingConfig class you need to prepend the parameter name with "training".

The trained model is saved in `outputs/models/<problem><problem_dim>-<training.method>`.

### Evaluation
Evaluation with the same settings used in the paper can be done with `eval.py` script. As before, this can be configured with Hydra, for instance:

```bash
python scripts/eval.py +experiment=knapsack_ppo
```

The `eval.py` script already sweeps over the different number of steps (`sa.outer_steps`) considered in paper. It also runs vanilla Simulated Annealing and a greedy variant of Neural SA. The results are stored in `outputs/results/<problem>` folder, and can be aggregated and printed with the `print_results.py` script:

```bash
python scripts/print_results.py +experiment=knapsack_ppo
```
