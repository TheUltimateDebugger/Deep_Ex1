import os
import torch
from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): path to the folder containing the peptide .txt files
        """
        super().__init__()
        self.folder_path = folder_path
        self.peptides = []
        self.labels = []
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}

        self.label_mapping = {}  # filename base -> label index
        self._load_data()

    def _load_data(self):
        """
        Private method to load data from the folder_path.
        """
        allele_files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')]

        current_label = 0
        for filename in sorted(allele_files):
            filepath = os.path.join(self.folder_path, filename)

            # Remove the .txt and the '_pos' or '_neg' suffix if necessary
            base_name = filename.replace('.txt', '')

            if base_name not in self.label_mapping:
                self.label_mapping[base_name] = current_label
                current_label += 1

            label = self.label_mapping[base_name]

            with open(filepath, 'r') as file:
                lines = file.read().splitlines()
                for line in lines:
                    if line.strip():
                        self.peptides.append(line.strip())
                        self.labels.append(label)

        print(f"Loaded {len(self.peptides)} peptides and {len(self.labels)} labels")
        print(f"Label mapping: {self.label_mapping}")

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        label = self.labels[idx]

        # Convert peptide into one-hot tensor
        one_hot = torch.zeros(len(peptide), len(self.amino_acids))
        for i, aa in enumerate(peptide):
            aa_idx = self.aa_to_idx.get(aa, None)
            if aa_idx is not None:
                one_hot[i, aa_idx] = 1.0
            else:
                raise ValueError(f"Unknown amino acid '{aa}' in peptide '{peptide}'")

        one_hot = one_hot.view(-1)  # Flatten from (length, 20) to (length*20,)
        return one_hot, torch.tensor(label, dtype=torch.long)
