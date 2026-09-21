"""Expand experiment matrices into explicit configs and commands without running."""
import copy
import itertools

from .io import portable, read_json, resolve, write_json


def plan(spec, output):
    directory = resolve(output)
    directory.mkdir(parents=True, exist_ok=False)
    commands, jobs = [], []
    for experiment in spec["experiments"]:
        base = read_json(experiment["base_config"])
        grid = experiment.get("grid", {})
        keys = list(grid)
        combinations = itertools.product(*(grid[k] for k in keys))
        for i, values in enumerate(combinations):
            config = copy.deepcopy(base)
            for key, value in zip(keys, values):
                current = config
                parts = key.split(".")
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            name = "{}_{:03d}".format(experiment["name"], i)
            path = directory / (name + ".json")
            write_json(path, config)
            command = ["python", "-m", "riverlab", experiment["command"],
                       "--config", portable(path), "--output", "runs/"+name, "--device", "cuda"]
            commands.append(" ".join('"{}"'.format(x) if " " in x else x for x in command))
            jobs.append({"name": name, "command": command, "config": config})
    (directory / "commands.txt").write_text("\n".join(commands)+"\n", encoding="utf-8")
    write_json(directory / "jobs.json", jobs)
    return {"jobs": len(jobs), "note": "Plan only. Commands have not been executed."}
