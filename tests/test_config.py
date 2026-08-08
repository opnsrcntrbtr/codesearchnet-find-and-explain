"""Config loading tests.

A silently-ignored hyperparameter would invalidate an experiment without any
visible signal, so most of these assert that bad configs fail loudly.
"""

from __future__ import annotations

import pytest

from csne.config import load_experiment, load_yaml

EXPERIMENT = """
name: demo
data: ../data.yaml
retrieval: ../retrieval.yaml
summarization: ../summarization.yaml
evaluation: ../eval.yaml
overrides:
  retrieval:
    epochs: 7
"""


@pytest.fixture
def config_tree(tmp_path):
    """A miniature `configs/` tree with shared files and one experiment."""
    (tmp_path / "data.yaml").write_text("language: python\nmax_eval: 50\n")
    (tmp_path / "retrieval.yaml").write_text("epochs: 1\nbatch_size: 8\n")
    (tmp_path / "summarization.yaml").write_text("backend: hf\n")
    (tmp_path / "eval.yaml").write_text("retrieval_k: [1, 5]\n")
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "demo.yaml").write_text(EXPERIMENT)
    return experiments / "demo.yaml"


def test_loads_shared_sections_by_path(config_tree):
    """Sections given as paths resolve relative to the experiment file."""
    config = load_experiment(config_tree)
    assert config.name == "demo"
    assert config.data.max_eval == 50
    assert config.summarization.backend == "hf"
    assert config.evaluation.retrieval_k == [1, 5]


def test_overrides_win_over_shared_files(config_tree):
    """That is the whole point of `overrides` — experiments differ by a key."""
    assert load_experiment(config_tree).retrieval.epochs == 7


def test_unset_keys_fall_back_to_defaults(config_tree):
    """A shared file need not restate every field."""
    assert load_experiment(config_tree).retrieval.batch_size == 8
    assert load_experiment(config_tree).retrieval.max_seq_length == 256


def test_unknown_key_is_rejected(tmp_path):
    """A typo'd hyperparameter must fail, not silently do nothing."""
    (tmp_path / "e.yaml").write_text("name: x\nretrieval:\n  epocs: 3\n")
    with pytest.raises(ValueError, match="unknown RetrievalConfig key"):
        load_experiment(tmp_path / "e.yaml")


def test_unknown_override_section_is_rejected(tmp_path):
    (tmp_path / "e.yaml").write_text("name: x\noverrides:\n  retrievel:\n    epochs: 3\n")
    with pytest.raises(ValueError, match="unknown override section"):
        load_experiment(tmp_path / "e.yaml")


def test_missing_name_is_rejected(tmp_path):
    """Results rows are keyed by run name; an unnamed experiment is untraceable."""
    (tmp_path / "e.yaml").write_text("retrieval:\n  epochs: 3\n")
    with pytest.raises(ValueError, match="must define a 'name'"):
        load_experiment(tmp_path / "e.yaml")


def test_load_yaml_rejects_non_mapping(tmp_path):
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_yaml(tmp_path / "list.yaml")


def test_load_yaml_treats_empty_file_as_empty_mapping(tmp_path):
    (tmp_path / "empty.yaml").write_text("")
    assert load_yaml(tmp_path / "empty.yaml") == {}


def test_shipped_experiment_configs_load():
    """The configs committed to the repo must actually parse."""
    for name in ("baseline", "finetune_minilm"):
        config = load_experiment(f"configs/experiments/{name}.yaml")
        assert config.data.retrieval_dataset == "code-search-net/code_search_net"
        assert config.data.retrieval_config == "python"


def test_baseline_config_does_not_train():
    """`epochs: 0` is how the baseline evaluates the untouched base encoder."""
    assert load_experiment("configs/experiments/baseline.yaml").retrieval.epochs == 0
