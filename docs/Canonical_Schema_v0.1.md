# Canonical Schema:

{
  "event_id": "uuid-string",
  "sensor_type": "acoustic" | "vision" | "hybrid",
  "ts_ns": 1234567890123456,
  "rx_ns": 1234567891123456,
  "latency_ns": 1000,
  "latency_status": "normal" | "delayed" | "obsolete",
  "location": {
    "lat": 52.520000,
    "lon": 13.405000,
    "error_radius_m": 30.0
  },
  "bearing_deg": 235.0,
  "bearing_confidence": 0.78,
  "n_objects_detected": 1,
  "event_code": "DRONE_DET",
  "sensor_node_id": "NODE_A04",
  "unit_id": null,
  "unit_id_status": "unknown",
  "equipment_type": null,
  "equipment_type_status": "unknown",
  "equipment_type_confidence": 0.0,
  "equipment_candidate_list": [],
  "optional_activity_code": null,
  "sensor_metadata": {
    "sensor_deployment_lat": 52.5205,
    "sensor_deployment_lon": 13.4055,
    "sensor_orientation_azimuth_deg": 120.0,
    "sensor_orientation_error_deg": 5.0,
    "sensor_health_status": "nominal",
    "microphone_array_config": null,
    "node_geometry_baseline_m": null
  },
  "classification_history": [
    {
      "timestamp_ns": 1234567890000000,
      "model_type": "acoustic_v1",
      "classification": "QuadCopter-X",
      "confidence": 0.45
    }
  ],
  "contributing_edges": [],
  "aggregated_location_lat": null,
  "aggregated_location_lon": null,
  "aggregation_confidence": null,
  "object_track_id": null,
  "location_method": "bearing_average" | "acoustic_triangulation" | "sensor_fusion",
  "remarks": "Initial acoustic detection, vision follow-up pending"
}

<!-- Updated Field Description:
Field Name	Type	Description	Notes
event_id	string	Unique identifier for the event (UUID)	Used for tracking and duplication control
sensor_type	enum string	Type of detecting sensor (“acoustic”, “vision”, “hybrid”)	Distinguishes sensor origin
ts_ns	uint64	Timestamp of detection in nanoseconds	Provides the “time” component
rx_ns	uint64	Timestamp of receipt in nanoseconds	Enables end-to-end latency measurement
latency_ns	uint64	Difference between receipt and detection timestamp (rx_ns - ts_ns)	Used for freshness evaluation
latency_status	enum string	Latency status (“normal”, “delayed”, “obsolete”)	Flags how current/fresh the information is
location.lat	float	Latitude (in degrees) of the event	Location component
location.lon	float	Longitude (in degrees) of the event	Location component
location.error_radius_m	float	Radius of location uncertainty (meters)	Adds confidence dimension (how precise the location is)
bearing_deg	float	Bearing angle (0-360°) from sensor to object	Important directional cue
bearing_confidence	float	Confidence of bearing measurement (0.0-1.0)	Indicates reliability of direction
n_objects_detected	int	Number of objects detected in this event	Reflects size or multiplicity
event_code	string	Code identifying type of event (e.g., “DRONE_DET”)	Categorization of event
sensor_node_id	string	Identifier of the sensor node that reported the event	Helps trace sensor origin
unit_id	string or null	Identified unit (if known)	“Unit identification” if available
unit_id_status	enum string	Status of unit identification (“unknown”, “assumed”, “classified”)	Indicates reliability of unit info
equipment_type	string or null	Type/model of equipment/object detected (if known)	Equipment/object classification
equipment_type_status	enum string	Status of equipment identification (“unknown”, “assumed”, “classified”)	Reliability of equipment info
equipment_type_confidence	float	Confidence of equipment type classification (0.0-1.0)	Helps weigh equipment claim
equipment_candidate_list	array	List of possible equipment types with confidences	Supports multi-hypothesis equipment classification
optional_activity_code	string or null	Code for detected entity activity/behaviour (if known)	Supports the “Activity” dimension if available
sensor_metadata.sensor_deployment_lat	float	Latitude of sensor node deployment	Helps map sensor geometry
sensor_metadata.sensor_deployment_lon	float	Longitude of sensor node deployment	Helps map sensor geometry
sensor_metadata.sensor_orientation_azimuth_deg	float	Orientation azimuth of sensor node (0-360°)	Influences bearing accuracy
sensor_metadata.sensor_orientation_error_deg	float	Error/uncertainty of sensor node orientation (degrees)	Adds meta-confidence dimension
sensor_metadata.sensor_health_status	enum string	Health status of sensor node (“nominal”, “degraded”, “failed”)	Helps assess data quality
sensor_metadata.microphone_array_config	object or null	Description/config for microphone array if present	Enables intra-node acoustic localisation if available
sensor_metadata.node_geometry_baseline_m	float or null	Baseline geometry in meters for node or mic array	Supports triangulation/geometry weighting
classification_history	array of objects	History log of classification attempts (timestamp, model_type, classification, confidence)	Enables traceability of classification changes
contributing_edges	array	List of contributing sensor nodes (sensor_node_id + metadata)	Captures multi-node aggregation footprint
aggregated_location_lat	float or null	Aggregated/fused location latitude (if multi-node)	Result of node aggregation
aggregated_location_lon	float or null	Aggregated/fused location longitude	Result of node aggregation
aggregation_confidence	float or null	Confidence of the aggregated/fused location (0.0-1.0)	Indicates reliability of fused result
object_track_id	string or null	Unique identifier for tracked object (across updates)	Enables persistent tracking of the same object
location_method	enum string	Method used to derive location (“bearing_average”, “acoustic_triangulation”, “sensor_fusion”)	Clarifies how the location was computed
remarks	string	Free-text remarks or supplementary info	Operational or domain notes -->
