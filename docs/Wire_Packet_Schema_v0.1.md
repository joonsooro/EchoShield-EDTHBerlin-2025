# Wire Packet Schema (edge schema): .proto (~40B, Durable in baseband transmission environment)
syntax = "proto3";
package EchoShield;

enum SensorType {
  SENSOR_UNKNOWN = 0;
  SENSOR_ACOUSTIC = 1;
  SENSOR_VISION = 2;
  SENSOR_HYBRID = 3;
}

enum LocationMethod {
  LOC_UNKNOWN = 0;
  LOC_BEARING_ONLY = 1;
  LOC_ACOUSTIC_TRIANGULATION = 2;
  LOC_SENSOR_FUSION = 3;
}

message LocationInt {
  int32 lat_int = 1;       // e.g., degrees * 1e5
  int32 lon_int = 2;
  uint16 error_radius_m = 3;
}

message ContributorInfo {
  string sensor_node_id = 1;
  int64 ts_ns = 2;
  uint16 bearing_deg = 3;
  uint8 bearing_confidence = 4;
}

message WirePacket {
  string event_id = 1;
  SensorType sensor_type = 2;
  uint64 ts_ns = 3;                 // detection timestamp
  uint64 rx_ns = 4;                 // reception timestamp
  LocationInt location = 5;
  uint16 bearing_deg = 6;           // 0-360°, scaled
  uint8 bearing_confidence = 7;     // 0-255 scale
  uint8 n_objects_detected = 8;
  uint8 event_code = 9;
  string sensor_node_id = 10;
  uint8 equipment_type_status = 11;   // e.g., 0=unknown,1=assumed,2=classified
  uint8 unit_id_status = 12;          // similar status code
  LocationMethod location_method = 13;  // NEW: method used by the node/packet
  repeated ContributorInfo contributing_edges = 14;  // NEW: optional list of nodes contributing
}
