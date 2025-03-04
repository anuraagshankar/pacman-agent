# Pacman Agent

A template for coding a pacman agent.

## Setting up the environment

1. Copy or clone the code from this framework to create your Pacman Agent, e.g., `https://github.com/anuraagshankar/pacman-agent.git`
2. Go into pacman-agent folder, `cd pacman-agent/`
3. Run `git submodule update --init --remote` to pull the last pacman-contest
4. Create a virtual environment, e.g., `python -m venv venv`
5. Activate the virtual environment with `source venv/bin/activate`
6. Run `pip install -r requirements.txt`
7. Go to the pacman-contest folder and install the requirements again:
   - `cd pacman-contest/`
   - `pip install -r requirements.txt`
   - `pip install -e .`
8. Run the sample capture.py file:
   - `cd src/contest/`
   - `python capture.py`

## Coding a new agent

In the root folder do the following:

1. Create in `my_team.py` a class with the name of your agent that inherits from `CaptureAgent`, e.g. `class ReflexCaptureAgent(CaptureAgent):`
2. In the new class, override the `def choose_action(self, game_state):` function to return the best next action (check the given source code example).
3. (Optional) Add any other functions to the class for reasoning / learning and improving your agents decision which could also use other code sources in the same folder.

To debug the agent you can run `capture.py` between the `baseline_team` and your current agent:

1. `cd pacman-contest/src/contest/`
2. `python capture.py -r baseline_team -b ../../../my_team.py`
