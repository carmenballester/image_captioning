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
    cap = datasets.CocoCaptions(root = args.val_image_path,
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
        # Crear un generador
        train_loader = load_data(args.images_path, args.captions_path, args.batch_size, stoi)

        epoch_losses = []
		# Get the images and captions for the current batch ?????
        for batch, (images, captions) in enumerate(train_loader):
            current_batch += 1

            # Load images to the GPU if it exists 
            images_v = images.to(dev)

			# Create embeddings for training.
			# Each word has a vectorial representation, so words with simmilar meanings have simmilar representations. 
            # The embeddings are the inputs to the model. 
            train_labels = [['<s>'] + label for label in captions]
            expected_labels = [label + ['</s>'] for label in captions]

            train_indices = [[stoi[word] for word in label] for label in train_labels]
            expected_indices = [[stoi[word] for word in label] for label in expected_labels]

            max_length = max([len(label) for label in train_indices])
            train_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in train_indices])
            expected_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in expected_indices])
           
			
            # Load embeddings to the GPU if it exists 
			train_indices_v = train_indices_v.to(dev)
            expected_indices_v = expected_indices_v.to(dev)

            # Genera la lista de indices validos para el entrenamiento, ya que muchas frases tendran
            # padding y los resultados de pasar por el padding no son validos
            valid_training_indices = []
            for i, label in enumerate(train_labels):
                valid_training_indices = valid_training_indices + [j for j in range(i*(max_length), i*(max_length) + len(label))]

            # Desenrolla las salidas y las guarda en un tensor
            # flat_expected_ids = [i for ids in expected_ids for i in ids]
            # valid_expected_v = torch.LongTensor(expected_indices_v.view(-1)[valid_training_indices]).to(dev)
            valid_expected_v = expected_indices_v.view(-1)[valid_training_indices]

            # Limpia los optimizadores
            optimizer.zero_grad()

            # Calcula las predicciones
            outputs_v = cnn_cnn(images_v, train_indices_v)

            # Desenrolla las frases generadas para poder pasarlas por la funcion de perdida
            outputs_v = outputs_v.view(-1, cnn_cnn.vocab_size)

            # Calcula la pérdida para actualizar las redes
            loss = criterion(outputs_v[valid_training_indices], valid_expected_v)

            # Actualizar los pesos
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(cnn_cnn.parameters(), 1.0)

            # Paso del optimizador
            optimizer.step()

            epoch_losses.append(loss.item())

            # Informar sobre el estado del entrenamiento, cada 100 batches
            if batch % 500 == 0 and batch != 0:
                # if i % 100 == 0 and i != 0:
                # Actualizar tensorboard
                mean_batch_loss = np.mean(epoch_losses[-500:])
                writer.add_scalar('Training loss', mean_batch_loss, current_batch)

                # Imprimir status
                print('Epoch: {}\tBatch: {}\t\tTraining loss: {}'.format(e, batch, mean_batch_loss))

        else:
            print('--- Calculating BLEU score for epoch {}'.format(e))
            spell = SpellChecker(distance=1)
            score_bleu_1_epoch = 0
            score_bleu_2_epoch = 0
            score_bleu_3_epoch = 0
            score_bleu_4_epoch = 0
            for i in range(len(cap)):
            # for i in range(20):
                # Obtener una imagen con sus captions y cargarla en la tarjeta si hay una
                img, target = cap[i]
                img = img.to(dev).view(1, *img.shape)
                # Obtener el valor de precisión de la frase predicha utilizando BLEU
                sentence = cnn_cnn.sample(img, stoi, itos)[0]
                captions = [[word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", target[i].lower())] for i in range(len(target))]
                score_bleu_1_epoch += sentence_bleu(captions, sentence, weights=(1.0, 0.0, 0.0, 0.0))
                score_bleu_2_epoch += sentence_bleu(captions, sentence, weights=(0.5, 0.5, 0.0, 0.0))
                score_bleu_3_epoch += sentence_bleu(captions, sentence, weights=(0.333, 0.333, 0.333, 0.0))
                score_bleu_4_epoch += sentence_bleu(captions, sentence, weights=(0.25, 0.25, 0.25, 0.25))

            score_bleu_1.append(score_bleu_1_epoch / len(cap))
            score_bleu_2.append(score_bleu_2_epoch / len(cap))
            score_bleu_3.append(score_bleu_3_epoch / len(cap))
            score_bleu_4.append(score_bleu_4_epoch / len(cap))
            # score_history.append(score / 20)

            if (score_bleu_4[-1] == max(score_bleu_4)):
                print("Model saved!")
                cnn_cnn.save()

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
