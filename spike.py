import torch
import torch.nn.functional as F

from dataset import PeptideDataset
from mlp_class import MLPBetterClassifier
import pandas as pd

# Constants
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}


def one_hot_encode(peptide):
    one_hot = torch.zeros(len(peptide), len(amino_acids))
    for i, aa in enumerate(peptide):
        aa_idx = aa_to_idx.get(aa, None)
        if aa_idx is not None:
            one_hot[i, aa_idx] = 1.0
        else:
            raise ValueError(f"Unknown amino acid '{aa}'")
    return one_hot.view(-1).unsqueeze(0)  # shape: (1, L*20)

def generate_9mers(sequence):
    return [sequence[i:i+9] for i in range(len(sequence) - 8)]


def predict_peptides(peptides):
    results = []

    dataset = PeptideDataset("ex1 data\\ex1 data")
    idx_to_class = {v: k for k, v in dataset.label_mapping.items()}

    # Load model once
    model = MLPBetterClassifier()
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()

    for peptide in peptides:
        try:
            # One-hot encode input
            input_tensor = one_hot_encode(peptide.upper()).float().unsqueeze(0)  # add batch dim

            # Get model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred_class = int(torch.argmax(logits))

            results.append({
                "Peptide": peptide,
                "Predicted Class": idx_to_class[pred_class],
                "Probability (Positive)": probs[1] if len(probs) > 1 else probs[0]
            })
        except Exception as e:
            print(f"Error processing peptide {peptide}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by highest predicted positive probability
    df = df.sort_values(by="Probability (Positive)", ascending=False).reset_index(drop=True)

    # Print top 3
    print("Top 3 Predicted Peptides:")
    print(df.head(10))

    return df

def main(spike_file_path):
    """
    runs the code for part e of the exersice
    """
    with open(spike_file_path, 'r') as f:
        sequence = f.read().strip().replace('\n', '').replace(' ', '')

    peptides = generate_9mers(sequence)
    results_df = predict_peptides(peptides)
    results_df.to_csv("spike_predictions.csv", index=False)
    print("\nResults saved to spike_predictions.csv")

if __name__ == "__main__":
    main("ex1 data\\spike.txt")
