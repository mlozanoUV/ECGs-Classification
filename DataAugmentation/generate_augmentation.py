"""
    Copyright (C) 2020 - Guillermo Jimenez-Perez <guillermo@jimenezperez.com>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from argparse import ArgumentParser

from typing import Tuple

import os
import os.path
import sklearn.preprocessing
import math
import shutil
import pathlib
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import tqdm
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def mixup(X: np.ndarray, y: np.ndarray, ids: np.ndarray, N_max: int = 1, alpha: float = 5.0, beta: float = 1.5, include_same_patient: bool = False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Function takes as input: 
    * X: matrix of N samples by M descriptors
    * y: matrix of N samples containing the labels as text (e.g. "LCC-RCC" or "left").
    * ids: vector containing the patient label (to disambiguate same patient with different lead configurations), e.g. "ECG_AJ_4"

    Returns: 
    * the augmented input matrix Xhat
    * the augmented label matrix yhat 
    * a weights matrix of N samples representing the relative weight of the samples 
      w.r.t. each other (w = 100 for original samples and w ∈ [0,1] for augmented samples)
    * a 
    """

    # Determine type of label (Left-vs-Right or Site-of-Origin)
    unique_labels = np.unique(y)
    unique_ids = np.array(["_".join(k.split("_")[:-2]) for k in ids])

    if unique_labels[0].upper() in ["LV", "RV"]:
        label_type = "LeftRight"
        possible_matches = {
            "LV"         : np.array(["LV"]),
            "RV"         : np.array(["RV"])
        }
    else:
        label_type = "Sublocation"
        possible_matches = {
            "LCC"           : np.array(["LCC","LCC-RCC","RCC","Summit","Anteroseptal"]),
            "RCC"           : np.array(["RCC", "LCC-RCC", "LCC", "Posteroseptal", "AMC"]),
            "LCC-RCC"       : np.array(["LCC", "LCC-RCC", "RCC"]),
            "AMC"           : np.array(["AMC", "RCC", "Summit"]),
            "Anteroseptal"  : np.array(["Anteroseptal", "Posteroseptal", "LCC", "Summit", "RFW"]),
            "Posteroseptal" : np.array(["Anteroseptal", "Posteroseptal", "RCC", "RFW"]),
            "RFW"           : np.array(["RFW", "Anteroseptal", "Posteroseptal"]),
            "Summit"        : np.array(["AMC", "Summit", "LCC", "Anteroseptal"])
        }

    # Iterate over signals
    Xhat = []
    yhat = []
    weights = []
    idhat = []
    # Iterate over source sample
    for i,(sample1,label1,uid1) in enumerate(zip(X,y,ids)):
        # Store original sample1
        Xhat.append(sample1)
        yhat.append(label1)
        idhat.append(uid1)
        weights.append(100) # Value to clearly discern original samples from augmented ones

        # Filter same patient
        filter_same_patient = ("_".join(uid1.split("_")[:-2]) == unique_ids)
        
        # Select available choices
        available_choices = (possible_matches[y[i]] == y[:,None])
        available_choices[i,:] = False # Disable current choice
        if not include_same_patient:
            available_choices[filter_same_patient,:] = False # Disable current choice
        available_choices = [np.where(available_choices[:,loc])[0] for loc in range(available_choices.shape[1])]

        # Randomly select N_max elements for every available option
        selected_choices = [np.random.choice(choices,size=N_max) for choices in available_choices]

        for j in np.concatenate(selected_choices):
            sample2 = np.copy(X[j])
            label2  = np.copy(y[j])
            uid2    = np.copy(ids[j])

            # Generate mixup parameters, conditioned to specific pairs (label1, label2)
            if ((label1,label2) == ("RCC", "AMC") or (label1,label2) == ("AMC", "RCC") or 
                (label1,label2) == ("RFW", "Anteroseptal") or (label1,label2) == ("Anteroseptal", "RFW")):
                # If these combinations of labels apply, obtain a lambda that is much closer to label1 to avoid creating too different signals
                lmbda = np.random.beta(alpha*10,beta)
            else:
                # Otherwise, just draw from a beta distribution
                lmbda = np.random.beta(alpha,beta)

            # Select final labels according to neighbouring segments
            samplemixup = lmbda*sample1 + (1 - lmbda)*sample2
            labelmixup  = label1 if lmbda >= 0.5 else label2
            weightmixup = lmbda if lmbda >= 0.5 else 1-lmbda
            weightmixup = (weightmixup-0.5)/0.5 # Normalize in [0,1]
            uidmixup    = f"{uid1}###{uid2}###{j}"

            # Case-specific classifications (if using fine labels)
            if label_type == "Sublocation":
                if   (label1 == "LCC") and (label2 == "RCC"):
                    labelmixup = "LCC" if lmbda >= 1/3 else "LCC-RCC" if lmbda <= 2/3 else "RCC"
                elif (label1 == "RCC") and (label2 == "LCC"):
                    labelmixup = "RCC" if lmbda >= 1/3 else "LCC-RCC" if lmbda <= 2/3 else "LCC"

            # Generate mixed-up version of the QRS
            Xhat.append(samplemixup)
            yhat.append(labelmixup)
            idhat.append(uidmixup)
            weights.append(weightmixup)

    # As arrays
    Xhat    = np.array(Xhat)
    yhat    = np.array(yhat)
    idhat   = np.array(idhat)
    weights = np.array(weights)

    return Xhat,yhat,weights,idhat


def mixup_old(X: np.ndarray, y: np.ndarray, ids: np.ndarray, repetitions: int = 1000, alpha: float = 5.0, beta: float = 1.5, include_same_patient: bool = False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Function takes as input: 
    * X: matrix of N samples by M descriptors
    * y: matrix of N samples containing the labels as text (e.g. "LCC-RCC" or "left").
    * ids: vector containing the patient label (to disambiguate same patient with different lead configurations), e.g. "ECG_AJ_4"

    Returns: 
    * the augmented input matrix Xhat
    * the augmented label matrix yhat 
    * a weights matrix of N samples representing the relative weight of the samples 
      w.r.t. each other (w = 100 for original samples and w ∈ [0,1] for augmented samples)
    * a 
    """

    # Determine type of label (Left-vs-Right or Site-of-Origin)
    unique_labels = np.unique(y)
    if unique_labels[0].lower() in ["left", "right"]:
        label_type = "LeftRight"
    else:
        label_type = "Sublocation"

    # Iterate over signals
    Xhat = []
    yhat = []
    weights = []
    idhat = []
    # Iterate over source sample
    for i,(sample1,label1,uid1) in enumerate(zip(X,y,ids)):
        # Store original sample1
        Xhat.append(sample1)
        yhat.append(label1)
        idhat.append(uid1)
        weights.append(100) # Value to clearly discern original samples from augmented ones

        # Iterate over target sample
        for j,(sample2,label2,uid2) in enumerate(zip(X,y,ids)):
            # If same signal, skip (no mixup possible)
            if i == j: continue

            # If chosen to skip same input morphology, skip
            if (uid1 == uid2) and not include_same_patient: continue

            # If not neighbouring labels (case using fine labels), skip
            if label_type == "Sublocation":
                if (label1 == "LCC")           and (label2 not in ["LCC","LCC-RCC","RCC","Summit","Anteroseptal"]): continue
                if (label1 == "RCC")           and (label2 not in ["RCC", "LCC-RCC", "LCC", "Posteroseptal", "AMC"]): continue
                if (label1 == "LCC-RCC")       and (label2 not in ["LCC", "LCC-RCC", "RCC"]): continue
                if (label1 == "AMC")           and (label2 not in ["AMC", "RCC", "Summit"]): continue
                if (label1 == "Anteroseptal")  and (label2 not in ["Anteroseptal", "Posteroseptal", "LCC", "Summit", "RFW"]): continue
                if (label1 == "Posteroseptal") and (label2 not in ["Anteroseptal", "Posteroseptal", "RCC", "RFW"]): continue
                if (label1 == "RFW")           and (label2 not in ["RFW", "Anteroseptal", "Posteroseptal"]): continue
                if (label1 == "Summit")        and (label2 not in ["AMC", "Summit", "LCC", "Anteroseptal"]): continue
            
            for rep in range(repetitions):
                # Generate mixup parameters, conditioned to specific pairs (label1, label2)
                if ((label1,label2) == ("RCC", "AMC") or (label1,label2) == ("AMC", "RCC") or 
                    (label1,label2) == ("RFW", "Anteroseptal") or (label1,label2) == ("Anteroseptal", "RFW")):
                    # If these combinations of labels apply, obtain a lambda that is much closer to label1 to avoid creating too different signals
                    lmbda = np.random.beta(alpha*10,beta)
                else:
                    # Otherwise, just draw from a beta distribution
                    lmbda = np.random.beta(alpha,beta)

                # Select final labels according to neighbouring segments
                samplemixup = lmbda*sample1 + (1 - lmbda)*sample2
                labelmixup = label1 if lmbda >= 0.5 else label2
                weightmixup = lmbda if lmbda >= 0.5 else 1-lmbda
                weightmixup = (weightmixup-0.5)/0.5 # Normalize in [0,1]
                uidmixup = f"{uid1}###{uid2}###{rep}"

                # Case-specific classifications (if using fine labels)
                if label_type == "Sublocation":
                    if   (label1 == "LCC") and (label2 == "RCC"):
                        labelmixup = "LCC" if lmbda >= 1/3 else "LCC-RCC" if lmbda <= 2/3 else "RCC"
                    elif (label1 == "RCC") and (label2 == "LCC"):
                        labelmixup = "RCC" if lmbda >= 1/3 else "LCC-RCC" if lmbda <= 2/3 else "LCC"

                # Generate mixed-up version of the QRS
                Xhat.append(samplemixup)
                yhat.append(labelmixup)
                idhat.append(uidmixup)
                weights.append(weightmixup)

    # As arrays
    Xhat    = np.array(Xhat)
    yhat    = np.array(yhat)
    idhat   = np.array(idhat)
    weights = np.array(weights)

    return Xhat,yhat,weights,idhat


