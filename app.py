import streamlit as st
import torch
import torch.nn.functional as F
from mlp_class import MLPClassifier

# Constantst
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

# Load trained model
model = MLPClassifier()
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Peptide Classification Interface")
peptide_input = st.text_input("Enter a peptide sequence (length 9):", max_chars=9)

if st.button("Predict") and peptide_input:
    try:
        input_tensor = one_hot_encode(peptide_input.upper()).float()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
            pred_class = int(torch.argmax(logits))

        st.write(f"Predicted Class: {pred_class}")
        st.bar_chart(probs)

    except ValueError as e:
        st.error(str(e))
