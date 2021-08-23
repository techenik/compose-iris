import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {1: "Kama", 2: "Rosa", 3: "Canadian"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "area": d.area,
            "perimeter": d.perimeter,
            "compactness": d.compactness,
            "lengthOfKernel": d.lengthOfKernel,
            "widthOfKernel": d.widthOfKernel,
            "asymmetryCoefficient": d.asymmetryCoefficient,
            "lengthOfKernelGroove": d.lengthOfKernelGroove,
            "seedType": d.seedType,
        }
        for d in data
    ]

    return processed
