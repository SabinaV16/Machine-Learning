from PIL import Image
from skimage.feature import hog
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Redimensionare și conversie la scala de gri a imaginilor pentru antrenare și testare
# poza de testare
numar_imagini = 26
numar_acuratete = 0
for i in range(1, 27):
    img = Image.open(f"Test_{i}.jpeg")
    img_resized = img.resize((200, 200), resample=Image.LANCZOS)
    img_resized.save(f"Test{i}_mod.jpeg")

    # grayscale
    image = Image.open(f"Test{i}_mod.jpeg")
    image = image.convert("L")
    image.save(f"Test{i}_mod.jpeg")

# redimensionare imagini cu trandafiri
for i in range(1, 101):
    img = Image.open(f"Trandafir_{i}.jpeg")
    img_resized = img.resize((200, 200), resample=Image.LANCZOS)
    img_resized.save(f"Trandafir_{i}.jpeg")

    # grayscale
    image = Image.open(f"Trandafir_{i}.jpeg")
    image = image.convert("L")
    image.save(f"Trandafir_{i}.jpeg")

# redimensionare imagini cu lalele
for i in range(1, 101):
    img = Image.open(f"Lalea_{i}.jpeg")
    img_resized = img.resize((200, 200), resample=Image.LANCZOS)
    img_resized.save(f"Lalea_{i}.jpeg")

    # grayscale
    image = Image.open(f"Lalea_{i}.jpeg")
    image = image.convert("L")
    image.save(f"Lalea_{i}.jpeg")

# extragerea caracteristicilor HOG din imagininile de antrenare
def extrage_caracteristici_HOG_si_salveaza():
    lista_caracteristici = []
    etichete = []

    for categorie in ["Trandafir", "Lalea"]:
        numar_imagini = 100
        for i in range(1, numar_imagini + 1):
            img = Image.open(f"{categorie}_{i}.jpeg")

            valori, _ = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            lista_caracteristici.append(valori)
            etichete.append(0 if categorie == "Trandafir" else 1)

    # Convertim lista de caracteristici și etichete într-un DataFrame Pandas pentru datele de antrenare
    df_antrenare = pd.DataFrame(np.array(lista_caracteristici))
    df_antrenare['Eticheta'] = etichete

    # Salvăm DataFrame-ul pentru datele de antrenare într-un fișier CSV
    df_antrenare.to_csv("trandafiri_lalele_caracteristici_HOG_antrenare.csv", index=False)

extrage_caracteristici_HOG_si_salveaza()

# Citirea datelor pentru antrenare
df_antrenare = pd.read_csv('trandafiri_lalele_caracteristici_HOG_antrenare.csv')

# Separarea datelor pentru antrenare și validare
df_val_trandafiri = df_antrenare[df_antrenare['Eticheta'] == 1].iloc[:60]
df_val_lalele = df_antrenare[df_antrenare['Eticheta'] == 0].iloc[:60]

# Datele pentru antrenare
df_antrenare = df_antrenare.drop(df_val_trandafiri.index)
df_antrenare = df_antrenare.drop(df_val_lalele.index)

# Separarea datelor în vectori de caracteristici și etichete pentru antrenare și validare
X_train = df_antrenare.drop('Eticheta', axis=1).values
y_train = df_antrenare['Eticheta'].values

X_validare = pd.concat([df_val_trandafiri, df_val_lalele]).drop('Eticheta', axis=1).values
y_validare = pd.concat([df_val_trandafiri, df_val_lalele])['Eticheta'].values

# Antrenarea modelului K-Nearest Neighbors pe caracteristicile HOG
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predictii pe datele de validare cu KNN
y_val_predict_knn = knn_model.predict(X_validare)

# Calcularea matricei de confuzie pentru KNN
conf_matrix_knn = confusion_matrix(y_validare, y_val_predict_knn)

# Afișarea matricei de confuzie pentru KNN
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictii')
plt.ylabel('Valori reale')
plt.title('Matrice de confuzie pentru datele de validare (KNN)')
plt.show()

# Raport de clasificare pentru KNN
report_knn = classification_report(y_validare, y_val_predict_knn, digits=2)
lines_knn = report_knn.split('\n')
filtered_report_knn = '\n'.join(lines_knn[0:6])
print("Raport de clasificare pentru datele de validare (KNN):\n", filtered_report_knn)
y_test = []

for i in range(1, numar_imagini + 1):
    y_test.append(1 if i <= 13 else 0)

y_test = np.array(y_test)
X_test = []

for i in range(1, numar_imagini + 1):
    X_test.append(1 if i <= 13 else 0)

X_test = np.array(X_test)

# Predictii pe datele de test cu KNN
y_test_predict_knn = knn_model.predict(X_test)

# Calcularea matricei de confuzie pentru datele de test cu KNN
conf_matrix_knn_test = confusion_matrix(y_test, y_test_predict_knn)

# Afișarea matricei de confuzie pentru datele de test cu KNN
sns.heatmap(conf_matrix_knn_test, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictii')
plt.ylabel('Valori reale')
plt.title('Matrice de confuzie pentru datele de test (KNN)')
plt.show()

# Raport de clasificare pentru datele de test cu KNN
report_knn_test = classification_report(y_test, y_test_predict_knn, digits=2)
lines_knn_test = report_knn_test.split('\n')
filtered_report_knn_test = '\n'.join(lines_knn_test[0:6])
print("Raport de clasificare pentru datele de test (KNN):\n", filtered_report_knn_test)

# Antrenarea modelului Naive Bayes pe caracteristicile HOG
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predictii pe datele de validare cu Naive Bayes
y_val_predict_nb = nb_model.predict(X_validare)

# Calcularea matricei de confuzie pentru Naive Bayes
conf_matrix_nb = confusion_matrix(y_validare, y_val_predict_nb)

# Afișarea matricei de confuzie pentru Naive Bayes
sns.heatmap(conf_matrix_nb, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictii')
plt.ylabel('Valori reale')
plt.title('Matrice de confuzie pentru datele de validare (Naive Bayes)')
plt.show()

# Raport de clasificare pentru Naive Bayes
report_nb = classification_report(y_validare, y_val_predict_nb, digits=2)
lines_nb = report_nb.split('\n')
filtered_report_nb = '\n'.join(lines_nb[0:6])
print("Raport de clasificare pentru datele de validare (Naive Bayes):\n", filtered_report_nb)

# Predictii pe datele de test cu Naive Bayes
y_test_predict_nb = nb_model.predict(X_test)

# Calcularea matricei de confuzie pentru datele de test cu Naive Bayes
conf_matrix_nb_test = confusion_matrix(y_test, y_test_predict_nb)

# Afișarea matricei de confuzie pentru datele de test cu Naive Bayes
sns.heatmap(conf_matrix_nb_test, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictii')
plt.ylabel('Valori reale')
plt.title('Matrice de confuzie pentru datele de test (Naive Bayes)')
plt.show()

# Raport de clasificare pentru datele de test cu Naive Bayes
report_nb_test = classification_report(y_test, y_test_predict_nb, digits=2)
lines_nb_test = report_nb_test.split('\n')
filtered_report_nb_test = '\n'.join(lines_nb_test[0:6])
print("Raport de clasificare pentru datele de test (Naive Bayes):\n", filtered_report_nb_test)



# Afișarea imaginilor de test cu eticheta prezisă de SVM
for i in range(1, numar_imagini + 1):
    image_path = f'Test{i}_mod.jpeg'
    image_path1 = f'Test_{i}.jpeg'
    test_image = imread(image_path)
    test_image1 = imread(image_path1, as_gray=False)
    test_hog_features, _ = hog(test_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    test_hog_features = np.array(test_hog_features).reshape(1, -1)

    # Prezicerea etichetei pentru imaginea de test cu SVM
    test_prediction_svm = nb_model.predict(test_hog_features)

    # Afisarea imaginii de test și a rezultatului testării cu SVM
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image1)
    plt.title(f"NB pentru imaginea de test {i}:\n Predictie: {'Trandafir' if test_prediction_svm[0] == 1 else 'Lalea'}\n Real : {'Trandafir' if i <= 13 else 'Lalea'}")
    plt.axis('off')
    plt.show()

# Afișarea imaginilor de test cu eticheta prezisă de Random Forest
for i in range(1, numar_imagini + 1):
    image_path = f'Test{i}_mod.jpeg'
    image_path1 = f'Test_{i}.jpeg'
    test_image = imread(image_path)
    test_image1 = imread(image_path1, as_gray=False)
    test_hog_features, _ = hog(test_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    test_hog_features = np.array(test_hog_features).reshape(1, -1)

    # Prezicerea etichetei pentru imaginea de test cu Random Forest
    test_prediction_rf = knn_model.predict(test_hog_features)

    # Afisarea imaginii de test și a rezultatului testării cu Random Forest
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image1)
    plt.title(f"KNN pentru imaginea de test {i}:\n Predictie: {'Trandafir' if test_prediction_rf[0] == 1 else 'Lalea'}\n Real : {'Trandafir' if i <= 13 else 'Lalea'}")
    plt.axis('off')
    plt.show()