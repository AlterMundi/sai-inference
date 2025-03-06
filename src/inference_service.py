from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import io

# Initialize FastAPI app
app = FastAPI()

# Load processor with auto dtype and device_map
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

# Load model with float16 precision and move to cuda:0
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="float16",
    device_map="cuda:0"
)

# Set model to bfloat16 for inference
model.to(dtype=torch.bfloat16)

# Define the API endpoint for image analysis
@app.post("/v1/chat/completions")
async def analyze_image(prompt: str = Form(...), image: UploadFile = File(...)):
    # Read and process the image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # Prepare inputs for the model
    inputs = processor.process(images=[image], text=prompt)

    # Move inputs to the device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Convert images to bfloat16 if present
    if "images" in inputs:
        inputs["images"] = inputs["images"].to(torch.bfloat16)

    # Generate output using the model
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
        tokenizer=processor.tokenizer
    )

    # Decode generated tokens
    generated_tokens = output[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"response": generated_text}

# Define a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

