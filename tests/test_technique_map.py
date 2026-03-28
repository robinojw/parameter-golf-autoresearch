"""Tests for technique adjacency map — bootstrap, merge, and injection."""

import json
from pathlib import Path

from research.reflect import bootstrap_technique_map, merge_technique_updates
from research.inject import render_technique_tree


class TestBootstrapTechniqueMap:
    def test_creates_map_from_baseline_techniques(self, tmp_path):
        path = tmp_path / "technique_map.json"
        bootstrap_technique_map(technique_map_path=path)
        data = json.loads(path.read_text())
        assert "nodes" in data
        assert "edges" in data
        # Check some baseline techniques are present (normalized to snake_case)
        node_keys = set(data["nodes"].keys())
        assert any("int6" in k for k in node_keys)
        for node_data in data["nodes"].values():
            assert node_data["status"] == "proven"

    def test_does_not_overwrite_existing(self, tmp_path):
        path = tmp_path / "technique_map.json"
        existing = {"nodes": {"custom": {"status": "exploring", "best_bpb": 1.15, "experiments": 1}}, "edges": []}
        path.write_text(json.dumps(existing))
        bootstrap_technique_map(technique_map_path=path)
        data = json.loads(path.read_text())
        assert "custom" in data["nodes"]


class TestMergeTechniqueUpdates:
    def test_adds_new_node(self, tmp_path):
        path = tmp_path / "technique_map.json"
        path.write_text(json.dumps({"nodes": {}, "edges": []}))
        updates = [{"node": "int4_QAT", "status": "dead_end", "parent": "int6_QAT", "relation": "refinement"}]
        merge_technique_updates(updates, technique_map_path=path)
        data = json.loads(path.read_text())
        assert "int4_QAT" in data["nodes"]
        assert data["nodes"]["int4_QAT"]["status"] == "dead_end"

    def test_adds_edge(self, tmp_path):
        path = tmp_path / "technique_map.json"
        path.write_text(json.dumps({"nodes": {}, "edges": []}))
        updates = [{"node": "int4_QAT", "status": "dead_end", "parent": "int6_QAT", "relation": "refinement"}]
        merge_technique_updates(updates, technique_map_path=path)
        data = json.loads(path.read_text())
        assert len(data["edges"]) == 1
        assert data["edges"][0]["parent"] == "int6_QAT"
        assert data["edges"][0]["child"] == "int4_QAT"

    def test_updates_existing_node_status(self, tmp_path):
        path = tmp_path / "technique_map.json"
        path.write_text(json.dumps({
            "nodes": {"TTT": {"status": "exploring", "best_bpb": 1.13, "experiments": 1}},
            "edges": [],
        }))
        updates = [{"node": "TTT", "status": "proven", "parent": None, "relation": None}]
        merge_technique_updates(updates, technique_map_path=path)
        data = json.loads(path.read_text())
        assert data["nodes"]["TTT"]["status"] == "proven"

    def test_no_duplicate_edges(self, tmp_path):
        path = tmp_path / "technique_map.json"
        path.write_text(json.dumps({
            "nodes": {},
            "edges": [{"parent": "A", "child": "B", "relation": "refinement"}],
        }))
        updates = [{"node": "B", "status": "exploring", "parent": "A", "relation": "refinement"}]
        merge_technique_updates(updates, technique_map_path=path)
        data = json.loads(path.read_text())
        assert len(data["edges"]) == 1


class TestRenderTechniqueTree:
    def test_renders_tree_with_statuses(self):
        data = {
            "nodes": {
                "int6_QAT": {"status": "proven", "best_bpb": 1.1194, "experiments": 3},
                "int4_QAT": {"status": "dead_end", "best_bpb": None, "experiments": 2},
                "EMA": {"status": "proven", "best_bpb": 1.12, "experiments": 1},
            },
            "edges": [{"parent": "int6_QAT", "child": "int4_QAT", "relation": "refinement"}],
        }
        result = render_technique_tree(data)
        assert "[proven] int6_QAT" in result
        assert "[dead_end] int4_QAT" in result
        assert "[proven] EMA" in result

    def test_renders_empty_map(self):
        result = render_technique_tree({"nodes": {}, "edges": []})
        assert result == ""
