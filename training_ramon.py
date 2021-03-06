import random
import argparse
import re
import os.path as path
import sys

import numpy as np

from spellchecker import SpellChecker

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nltk.translate.bleu_score import sentence_bleu

import model

# Usar CUDA si esta disponible
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

## DATA ---------------------------------------------------------------
# Las imagenes tienen que ir de 0 a 1, con un tamanio de 224x224
# despues normalizadas con normalize =
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
# 0.224, 0.225]) porque la red convolucional se entreno con
# imagenes con este formato

def dataloader(image_folder, captions_file, batch_size, stoi):

    spell = SpellChecker(distance=1)

    #Definir el tensor para guardar las imagenes de un batch
    tensor_images = torch.zeros((batch_size, 3, 224, 224)).to(device)

    #Definir las transformaciones que se aplican a las imagenes
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])

    #Cargar el dataset
    cap = datasets.CocoCaptions(root = image_folder,
                                annFile = captions_file,
                                transform = transform)

    for batch in range(int( len(cap)/batch_size )):
    # for batch in range(15):
        captions = []
        for i in range(batch_size):
            #Obtener una imagen con sus captions y seleccionar uno al azar
            img, target = cap[batch*batch_size+i]

            # Pasar la frase a todo minusculas y separar signos de puntuacion
#            captions.append([word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", target[random.randrange(5)].lower())])

			for word in re.findall(r"[\w']+|[.,!?;]", target[random.randrange(5)].lower()):
				if word in stoi:
					captions.append(word)
				else
					captions.append(spell.correction(word))

            #Actualizar el tensor de imagenes
            tensor_images[i] = img

        yield tensor_images, captions

def get_dicts():
    if not path.exists('./weights/dicts'):
        print('ERROR: dicts not created, please run "make dicts"')
        sys.exit(-1)

    dicts = torch.load('./weights/dicts')
    return dicts['stoi'], dicts['itos']

# def validation_losses():

def main(args):

    # Cargar los diccionarios
    stoi, itos = get_dicts()

    # Cargar fotos y captions para entrenar el modelo
    trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size, stoi)

    # Mantiene un tensorboard con datos actualizados del entrenamiento
    writer = SummaryWriter(comment='CNN_CNN_HA_CE')

    # Crea los modelos a entrenar
    cnn_cnn = model.CNN_CNN_HA_CE(len(stoi), 300, n_layers=args.n_layers, train_cnn=True).to(device)

    # Si existe una red preentrenada, cargarla
    cnn_cnn.load()

    # algoritmo que actualizara los pesos de las redes
    optimizer = optim.Adam([
        {'params': cnn_cnn.language_module_att.parameters()},
        {'params': cnn_cnn.prediction_module.parameters()},
        {'params': cnn_cnn.vision_module.parameters(), 'lr': 1e-5, 'weight_decay': 0.5e-5}
    ], lr=1e-4, weight_decay=0.1e-5)

    # Funcion de perdida a usar: Entropia cruzada ya que los elementos a predecir (palabras)
    # son mutuamente exlusivos (solo se puede elegir una palabra)
    criterion = torch.nn.NLLLoss()

    # Set de validacion
    cap = datasets.CocoCaptions(root = args.val_image_folder,
                                annFile = args.val_captions_file,
                                transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])]))

    ## ------------------------------------------------------------
    ## ------------------------------------------------------------
    ## -------------- Bucle de entrenamiento ----------------------
    current_batch = 0
    score_bleu_1 = []
    score_bleu_2 = []
    score_bleu_3 = []
    score_bleu_4 = []
    for e in range(args.epochs):
        # Crear un generador
        trainloader = dataloader(args.image_folder, args.captions_file, args.batch_size, stoi)

        epoch_losses = []
        for batch, (images, captions) in enumerate(trainloader):
            current_batch += 1

            # Pasa las imagenes a la tarjeta grafica en caso de que haya una
            images_v = images.to(device)

            # Crear embeddings para entrenar. Entradas al modelo
            train_labels = [['<s>'] + label for label in captions]
            expected_labels = [label + ['</s>'] for label in captions]
            train_indices = [[stoi[word] for word in label] for label in train_labels]
            expected_indices = [[stoi[word] for word in label] for label in expected_labels]
            max_length = max([len(label) for label in train_indices])
            train_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in train_indices])
            expected_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=-1) for label in expected_indices])
            train_indices_v = train_indices_v.to(device)
            expected_indices_v = expected_indices_v.to(device)

            # Genera la lista de indices validos para el entrenamiento, ya que muchas frases tendran
            # padding y los resultados de pasar por el padding no son validos
            valid_training_indices = []
            for i, label in enumerate(train_labels):
                valid_training_indices = valid_training_indices + [j for j in range(i*(max_length), i*(max_length) + len(label))]

            # Desenrolla las salidas y las guarda en un tensor
            # flat_expected_ids = [i for ids in expected_ids for i in ids]
            # valid_expected_v = torch.LongTensor(expected_indices_v.view(-1)[valid_training_indices]).to(device)
            valid_expected_v = expected_indices_v.view(-1)[valid_training_indices]

            # Limpia los optimizadores
            optimizer.zero_grad()

            # Calcula las predicciones
            outputs_v = cnn_cnn(images_v, train_indices_v)

            # Desenrolla las frases generadas para poder pasarlas por la funcion de perdida
            outputs_v = outputs_v.view(-1, cnn_cnn.vocab_size)

            # Calcula la p??rdida para actualizar las redes
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
                img = img.to(device).view(1, *img.shape)
                # Obtener el valor de precisi??n de la frase predicha utilizando BLEU
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
    parser.add_argument("--batch_size", type=int, default=8, help="Tama??o del batch")
    parser.add_argument("--image_folder", type=str, help='Carpeta que contiene todas las imagenes')
    parser.add_argument("--val_image_folder", type=str, help='Carpeta que contiene todas las imagenes para validar el modelo')
    parser.add_argument("--captions_file", type=str, help='Archivo JSON que contiene las frases')
    parser.add_argument("--val_captions_file", type=str, help='Archivo JSON que contiene las frases para validar el modelo')

    args = parser.parse_args()
    main(args)
