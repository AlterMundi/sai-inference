#!/bin/bash
# SAI Inference - Quick image analysis tool
# Usage: ./sai-inference.sh [OPTIONS] <image_path_or_directory>
# Output: Annotated images saved to ./out/ directory

# Default parameters
CONFIDENCE=""
IOU=""
CAMERA_ID=""
DETECTION_CLASSES=""
LINE_WIDTH="3"
API_URL="http://localhost:8888/api/v1/infer"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        -i|--iou)
            IOU="$2"
            shift 2
            ;;
        -d|--detection-classes)
            DETECTION_CLASSES="$2"
            shift 2
            ;;
        -m|--camera-id)
            CAMERA_ID="$2"
            shift 2
            ;;
        -w|--line-width)
            LINE_WIDTH="$2"
            shift 2
            ;;
        -u|--url)
            API_URL="$2"
            shift 2
            ;;
        --health)
            # Check API health
            health_url="${API_URL%/infer}/health"
            echo "Checking API health at: $health_url"
            echo ""
            response=$(curl -s "$health_url")
            if [ $? -eq 0 ]; then
                echo "$response" | jq -r '
                    "Status: \(.status)",
                    "Model: \(.loaded_model_info.name) (v\(.loaded_model_info.version))",
                    "Device: \(.loaded_model_info.device)",
                    "Confidence: \(.loaded_model_info.confidence_threshold)",
                    "IoU: \(.loaded_model_info.iou_threshold)",
                    "Input Size: \(.loaded_model_info.input_size)px",
                    "CPU: \(.system_metrics.cpu_usage)%",
                    "Memory: \(.system_metrics.memory_usage)%"
                ' 2>/dev/null || echo "$response"
            else
                echo "❌ API not reachable"
                exit 1
            fi
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <image_path_or_directory>"
            echo "       $0 --health"
            echo ""
            echo "Options:"
            echo "  -c, --confidence FLOAT        Confidence threshold (0.0-1.0, default: 0.39)"
            echo "  -i, --iou FLOAT               IoU threshold (0.0-1.0, default: 0.1)"
            echo "  -d, --detection-classes LIST  Classes to detect (e.g., '0' for smoke, '1' for fire, '0,1' for both)"
            echo "  -m, --camera-id ID            Camera identifier for tracking"
            echo "  -w, --line-width INT          Bounding box line width (default: 3)"
            echo "  -u, --url URL                 API endpoint (default: http://localhost:8888/api/v1/infer)"
            echo "  --health                      Check API health status"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --health                                     # Check API status"
            echo "  $0 img-test/                                    # Use defaults"
            echo "  $0 -c 0.5 -i 0.4 img-test/                      # Custom thresholds"
            echo "  $0 -d 0 -m camera01 tests/images/smoke/         # Smoke-only, with camera ID"
            echo "  $0 -c 0.3 -w 5 image.jpg                        # Low confidence, thick lines"
            echo ""
            echo "Output images with detections will be saved to ./out/"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            INPUT_PATH="$1"
            shift
            ;;
    esac
done

if [ -z "$INPUT_PATH" ]; then
    echo "Error: No image path or directory specified"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Create output directory
mkdir -p out

# Build curl parameters
CURL_PARAMS="-F return_image=true -F show_labels=true -F show_confidence=true -F line_width=$LINE_WIDTH"
[ -n "$CONFIDENCE" ] && CURL_PARAMS="$CURL_PARAMS -F confidence_threshold=$CONFIDENCE"
[ -n "$IOU" ] && CURL_PARAMS="$CURL_PARAMS -F iou_threshold=$IOU"
[ -n "$CAMERA_ID" ] && CURL_PARAMS="$CURL_PARAMS -F camera_id=$CAMERA_ID"

# Format detection_classes as JSON array if provided
if [ -n "$DETECTION_CLASSES" ]; then
    # If already has brackets, use as-is; otherwise wrap in brackets
    if [[ "$DETECTION_CLASSES" =~ ^\[.*\]$ ]]; then
        CURL_PARAMS="$CURL_PARAMS -F detection_classes=$DETECTION_CLASSES"
    else
        CURL_PARAMS="$CURL_PARAMS -F detection_classes=[$DETECTION_CLASSES]"
    fi
fi

if [ -d "$INPUT_PATH" ]; then
    # Directory: process all images
    echo "Processing images from: $INPUT_PATH"
    echo "Output directory: ./out/"
    [ -n "$CONFIDENCE" ] && echo "Confidence threshold: $CONFIDENCE"
    [ -n "$IOU" ] && echo "IoU threshold: $IOU"
    [ -n "$DETECTION_CLASSES" ] && echo "Detection classes: $DETECTION_CLASSES"
    [ -n "$CAMERA_ID" ] && echo "Camera ID: $CAMERA_ID"
    echo ""

    find "$INPUT_PATH" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | while read -r img; do
        filename=$(basename "$img")
        echo "Processing: $filename"

        # Call API with parameters
        response=$(curl -s -X POST "$API_URL" -F "file=@$img" $CURL_PARAMS)

        # Extract and decode base64 image if present
        annotated_image=$(echo "$response" | jq -r '.annotated_image // empty')
        if [ -n "$annotated_image" ]; then
            echo "$annotated_image" | base64 -d > "out/$filename"
            echo "  Saved to: out/$filename"
        else
            echo "  Warning: No annotated image returned"
        fi

        # Show detection results
        echo "$response" | jq -r 'if .detections | length > 0 then .detections[] | "  [\(.class_name)] confidence: \(.confidence | tonumber * 100 | round)%" else "  No detections" end'
        echo "---"
    done

    echo ""
    echo "✓ Processing complete. Check ./out/ for annotated images."

elif [ -f "$INPUT_PATH" ]; then
    # Single file
    filename=$(basename "$INPUT_PATH")
    echo "Processing: $filename"
    [ -n "$CONFIDENCE" ] && echo "Confidence threshold: $CONFIDENCE"
    [ -n "$IOU" ] && echo "IoU threshold: $IOU"
    [ -n "$DETECTION_CLASSES" ] && echo "Detection classes: $DETECTION_CLASSES"
    [ -n "$CAMERA_ID" ] && echo "Camera ID: $CAMERA_ID"
    echo ""

    # Call API with parameters
    response=$(curl -s -X POST "$API_URL" -F "file=@$INPUT_PATH" $CURL_PARAMS)

    # Extract and decode base64 image if present
    annotated_image=$(echo "$response" | jq -r '.annotated_image // empty')
    if [ -n "$annotated_image" ]; then
        echo "$annotated_image" | base64 -d > "out/$filename"
        echo "✓ Saved annotated image to: out/$filename"
    else
        echo "Warning: No annotated image returned"
    fi

    # Show detection results
    echo "$response" | jq -r 'if .detections | length > 0 then .detections[] | "[\(.class_name)] confidence: \(.confidence | tonumber * 100 | round)%" else "No detections" end'
    echo ""

else
    echo "Error: $INPUT_PATH is not a valid file or directory"
    exit 1
fi
