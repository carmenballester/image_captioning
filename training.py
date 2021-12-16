import argparse
import model
import numpy as np
import os.path as path
import random
import re
import sys
import torch
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu
from spellchecker import SpellChecker
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# Check CUDA availability
dev = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# DATA -----------------------------------------------------------------------------------------------------------
# Transform the dataset pictures to match the used training pictures 
# 1. Normalize pictures (0,1)
# 2. Resize (224x224)

def load_data(images_path, captions_path, batch_size, stoi):
    """
    Funtion for load the COCO dataset. 
    """

    # Inicialize ortographic corrector 
    spell = SpellChecker(distance=1) 

    # Define matrix for saving batch images and load to device
    mat_images = torch.zeros((batch_size, 3, 224, 224)).to(dev)

    # Define transformation 
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])

    # Load dataset
    data = datasets.CocoCaptions(root = images_path,
                                annFile = captions_path,
                                transform = transform)

    # Get the current batch 
    for batch in range(int( len(data)/batch_size )):
        captions = []
        # For each data in the current batch...
        for i in range(batch_size):
            # Get one image and its captions
            img, target = data[batch*batch_size+i]

            #captions.append([word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", target[random.randrange(5)].lower())])
            
            # Get one random caption and for each word in it...
            for word in re.findall(r"[\w']+|[.,!?;]", target[random.randrange(5)].lower()):
                # If word is in generated dictionary, add it to captions 
                if word in stoi: 
                    captions.append(word)
                
                # If not, correct it before adding 
                else: 
                    captions.append(spell.correction(word))
            
            # Add the loaded image 
            mat_images[i] = img

        yield mat_images, captions

def get_dicts():
    """
    Funtion for load the generated dictionary. 
    """

    if not path.exists('./weights/dicts'):
        print('ERROR: dicts not created, please run "make dicts"')
        sys.exit(-1)

    dicts = torch.load('./weights/dicts')
    return dicts['stoi'], dicts['itos']

# TRAINING ----------------------------------------------------------------------------------------------------

def main(args):

    # Load dictionary
    stoi, itos = get_dicts()

    # Load images and captions for training 
    train_loader = load_data(args.images_path, args.captions_path, args.batch_size, stoi)

    # Define tensorboard to visualize training data
    writer = SummaryWriter(comment='CNN_CNN_HA_CE')

    # Load neural network model
    cnn_cnn = model.CNN_CNN_HA_CE(len(stoi), 300, n_layers=args.n_layers, train_cnn=True).to(dev)

    # Load pretrained neural network if it exists 
    cnn_cnn.load()

    # Define optimizer for training stage 

	# LEARNING RATE HACE UNA COSA RARA	
    optimizer = optim.Adam([
        {'params': cnn_cnn.language_module_att.parameters()},
        {'params': cnn_cnn.prediction_module.parameters()},
        {'params': cnn_cnn.vision_module.parameters(), 'lr': 1e-5, 'weight_decay': 0.5e-5}], 
		lr=1e-4, weight_decay=0.1e-5)

	# Define loss function. Cross Entropy because the prediction is just one word (Good for classification problems)
    criterion = torch.nn.NLLLoss()

    # Define transformation 
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])

    # Define validation set
    validation_data = datasets.CocoCaptions(root = args.val_image_path,
                                annFile = args.val_captions_path,
                                transform = transform)

	# TRAINING LOOP -------------------------------------------------------------------------------------------
    current_batch = 0
    score_bleu_1 = []
    score_bleu_2 = []
    score_bleu_3 = []
    score_bleu_4 = []

	# For each epoch in training...
    for e in range(args.epochs):
        # Load the data from dataset
        train_loader = load_data(args.images_path, args.captions_path, args.batch_size, stoi)

        epoch_losses = []
		# Get the images and captions for the current batch ?????
        for batch, (images, captions) in enumerate(train_loader):
            current_batch += 1

            # Load images to the GPU if it exists 
            images = images.to(dev)

			# CREATE embeddings for training.
			# -------------------------------
			# Each word has a vectorial representation, so words with simmilar meanings have simmilar representations. 
            # The embeddings are the inputs to the model. 

			# Add the begining and the ending character acording to the documentation
            train_labels = [['<s>'] + label for label in captions]
            expected_labels = [label + ['</s>'] for label in captions]

			# Get the index for each word in each label
            train_ind = [[stoi[word] for word in label] for label in train_labels]
            expected_ind = [[stoi[word] for word in label] for label in expected_labels]

			# Resize all the captions index using padding so they are the same len
            max_length = max([len(label) for label in train_ind])
            train_ind_pad = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in train_ind])
            expected_ind_pad = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in expected_ind])
			
            # Load embeddings to the GPU if it exists 
			train_ind_pad = train_ind_pad.to(dev)
            expected_ind_pad = expected_ind_pad.to(dev)

			# Generate valid indices for training, removing padding
            train_ind_vect = []
            for i, label in enumerate(train_labels):
                train_ind_vect = train_ind_vect + [j for j in range(i*(max_length), i*(max_length) + len(label))]

            expected_ind_valid = expected_ind_pad.view(-1)[train_ind_vect]

			# GET predictions and UPDATE weights
			# ----------------------------------
            # Clean the optimizer
            optimizer.zero_grad()

            # Get the CNN prediction
            outputs = cnn_cnn(images, train_ind_pad)

            # Resize output for calculating loss function
            outputs = outputs.view(-1, cnn_cnn.vocab_size)

            # Calculate loss function without padding
            loss = criterion(outputs[train_ind_vect], expected_ind_valid)

            # Update the weights
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(cnn_cnn.parameters(), 1.0)

            # Optimizer step
            optimizer.step()

            epoch_losses.append(loss.item())

            # Show the training state
            if batch % 500 == 0 and batch != 0:
                # Calculate the mean loss in this batch
				mean_batch_loss = np.mean(epoch_losses[-500:])
                
				# Update tensorboard
                writer.add_scalar('Training loss', mean_batch_loss, current_batch)

                # Imprimir status
                print('Epoch: {}\tBatch: {}\t\tTraining loss: {}'.format(e, batch, mean_batch_loss))

		# CALCULATE BLEU score for this epoch
        else:
            print('--- Calculating BLEU score for epoch {}'.format(e))
            spell = SpellChecker(distance=1)
            score_bleu_1_epoch = 0
            score_bleu_2_epoch = 0
            score_bleu_3_epoch = 0
            score_bleu_4_epoch = 0
            for i in range(len(cap)):
                # Get one image and its caption and load it to the GPU if exists
                img, target = validation_data[i]
                img = img.to(dev).view(1, *img.shape)

                # Calculate the precision of the predicted phrase using BLEU
                sentence = cnn_cnn.sample(img, stoi, itos)[0]

                #captions = [[word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", target[i].lower())] for i in range(len(target))]

				for i in range(len(target)):
		            # Get each caption for the image and for each word in it...
		            for word in re.findall(r"[\w']+|[.,!?;]", target[i].lower()):
		                # If word is in generated dictionary, add it to captions 
		                if word in stoi: 
		                    captions.append(word)
		                
		                # If not, correct it before adding 
		                else: 
		                    captions.append(spell.correction(word))

	           	# Compare the tho phrases in differents word groups (see doc for further information) 
				# Sum of all the images in the validation dataset
                score_bleu_1_epoch += sentence_bleu(captions, sentence, weights=(1.0, 0.0, 0.0, 0.0))
                score_bleu_2_epoch += sentence_bleu(captions, sentence, weights=(0.5, 0.5, 0.0, 0.0))
                score_bleu_3_epoch += sentence_bleu(captions, sentence, weights=(0.333, 0.333, 0.333, 0.0))
                score_bleu_4_epoch += sentence_bleu(captions, sentence, weights=(0.25, 0.25, 0.25, 0.25))

			# Add the epoch mean BLUE score to the global BLUE score vector
            score_bleu_1.append(score_bleu_1_epoch/len(validation_data))
            score_bleu_2.append(score_bleu_2_epoch/len(validation_data))
            score_bleu_3.append(score_bleu_3_epoch/len(validation_data))
            score_bleu_4.append(score_bleu_4_epoch/len(validation_data))

			# Check if training is improving the performance
			# If not, save the trained model
            if (score_bleu_4[-1] == max(score_bleu_4)):
                print("Model saved!")
                cnn_cnn.save()

			# Show information in tensorboard
            writer.add_scalar('BLEU-1', score_bleu_1[-1], e)
            writer.add_scalar('BLEU-2', score_bleu_2[-1], e)
            writer.add_scalar('BLEU-3', score_bleu_3[-1], e)
            writer.add_scalar('BLEU-4', score_bleu_4[-1], e)
            writer.add_scalar('Epoch loss', np.mean(epoch_losses), e)

            print('--- BLEU-4 score: {}'.format(score_bleu_4[-1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--epochs", type=int, default=1, help="Numero de generaciones a entrenar")
    parser.add_argument("--n_layers", type=int, default=6, help='Capas del modulo del lenguaje')
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del batch")
    parser.add_argument("--image_folder", type=str, help='Carpeta que contiene todas las imagenes')
    parser.add_argument("--val_image_folder", type=str, help='Carpeta que contiene todas las imagenes para validar el modelo')
    parser.add_argument("--captions_file", type=str, help='Archivo JSON que contiene las frases')
    parser.add_argument("--val_captions_file", type=str, help='Archivo JSON que contiene las frases para validar el modelo')

    args = parser.parse_args()
    main(args)
