#!/usr/bin/env bash
# Example calls against a locally running API (make api)
curl -s localhost:8000/health; echo
curl -s -X POST "localhost:8000/predict?explain=true" -H 'Content-Type: application/json' \
  -d '{"machine_id":"demo-001","machine_type":"M","air_temperature_k":300.1,"process_temperature_k":310.2,"rotational_speed_rpm":1500,"torque_nm":42.5,"tool_wear_min":120}'; echo
