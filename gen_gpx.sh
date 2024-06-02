#!/bin/bash
mkdir gpx
python scripts/uavf_to_gpx.py uavf_2024/gnc/data/ARC/UPPER_FIELD_MISSION uavf_2024/gnc/data/ARC/UPPER_FIELD_DROPZONE gpx/arc_upper_field.gpx
python scripts/uavf_to_gpx.py uavf_2024/gnc/data/ARC/MAIN_FIELD/MISSION uavf_2024/gnc/data/ARC/MAIN_FIELD/AIRDROP_BOUNDARY gpx/arc_main_field.gpx
python scripts/uavf_to_gpx.py uavf_2024/gnc/data/ARC/CLUB_FIELD/MISSION uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY gpx/arc_club_field.gpx