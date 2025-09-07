#!/bin/bash
#
# SAI Inference Service - n8n Integration Test
# Tests the service with n8n webhook using verified image with known detections
#

set -e

# Configuration
TEST_IMAGE="/mnt/n8n-data/SAINet/predicts/img/sdis-07_brison-200_2024-02-02T11-24-22.jpg"
N8N_WEBHOOK="https://n8n.altermundi.net/webhook-test/e861ad7c-8160-4964-8953-5e3a02657293"
BEARER_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Im44bldlYmhvb2tBY2Nlc3MiLCJpYXQiOjE2NzY0MDI4MDAsImV4cCI6MTcwNzk2MDgwMCwiaXNzIjoibjhuQXBpIiwiYXVkIjoid2ViaG9va0NsaWVudCJ9.HKt6HB1KChxEXUusXBFrFupyxUhr0C2WW5IEfKwYZnw"

echo "🔥 SAI Inference Service - n8n Integration Test"
echo "================================================="
echo "Image: $(basename "$TEST_IMAGE")"
echo "Expected: 4 detections (1 fire, 3 smoke)"
echo "Webhook: $N8N_WEBHOOK"
echo ""

# Check if test image exists
if [[ ! -f "$TEST_IMAGE" ]]; then
    echo "❌ Test image not found: $TEST_IMAGE"
    exit 1
fi

# Prepare metadata JSON
METADATA=$(cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "sai_inference_test",
  "camera_id": "test_camera_001",
  "location": "test_site",
  "confidence_threshold": 0.15,
  "expected_detections": {
    "fire": 1,
    "smoke": 3,
    "total": 4
  },
  "image_name": "$(basename "$TEST_IMAGE")",
  "test_purpose": "n8n_integration_validation"
}
EOF
)

echo "🚀 Sending test image to n8n webhook..."

# Send request with curl
curl -X POST \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -F "image=@$TEST_IMAGE;type=image/jpeg" \
  -F "metadata=$METADATA;type=application/json" \
  -w "\n📡 HTTP Status: %{http_code}\n⏱️  Total Time: %{time_total}s\n📊 Upload Speed: %{speed_upload} bytes/sec\n" \
  "$N8N_WEBHOOK"

echo ""
echo "✅ n8n webhook test completed!"
echo "Check your n8n workflow execution for inference results."