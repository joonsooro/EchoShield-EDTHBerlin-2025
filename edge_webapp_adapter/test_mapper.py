from edge_webapp_adapter.mappers import to_wirepacket


def test_mapper_minimal():
    payload = {"nodeId":"EDGE01","time_ms":1730182040123,"confidence":0.81,"event":"drone"}
    w = to_wirepacket(payload)
    assert w["sensor_type"] == "acoustic"
    assert w["bearing_confidence"] == 81
    assert w["event_id"]
