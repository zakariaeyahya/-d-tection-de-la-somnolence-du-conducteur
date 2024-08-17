<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<img src="https://github.com/user-attachments/assets/c2239ac8-9c0b-434f-a47f-dfc473fbf5e1" alt="Image de détection de somnolence">

<h1>Détection de somnolence utilisant un Autoencoder Spatiotemporel</h1>

<p>
    <a href="mailto:zakariae.yh@gmail.com">zakariae.yh@gmail.com</a> |
    <a href="https://www.linkedin.com/in/zakariae-yahya">www.linkedin.com/in/zakariae-yahya</a>
</p>

<h2>Technologies utilisées</h2>
<p>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
</p>

<p>Ce projet Python se concentre sur la construction d'un système de détection de somnolence utilisant un Autoencoder Spatiotemporel. Le système traite les images de webcam, applique un modèle d'apprentissage profond pour classifier si les yeux de la personne sont ouverts ou fermés, et alerte l'utilisateur lorsque des signes de somnolence sont détectés.</p>

<h2>Structure du projet</h2>
<ul>
    <li><strong>train.py</strong> : Script pour entraîner le modèle Autoencoder Spatiotemporel. Il charge les images vidéo, prétraite les images, construit l'architecture du modèle et entraîne le modèle à détecter les anomalies.</li>
    <li><strong>test.py</strong> : Script pour la détection de somnolence en temps réel. Il utilise le modèle pré-entraîné pour classifier les états des yeux dans un flux vidéo en direct, détectant et notifiant les événements anormaux.</li>
</ul>

<h2>Instructions</h2>

<h3>Entraînement du modèle</h3>
<p>Exécutez train.py pour capturer des images d'une vidéo, prétraiter les images et entraîner le modèle Autoencoder Spatiotemporel.</p>
<pre><code>python train.py</code></pre>
<p>Le modèle entraîné sera sauvegardé sous le nom "saved_model.h5".</p>

<h3>Détection de somnolence en temps réel</h3>
<p>Remplacez __path_to_custom_test_video dans test.py par le chemin vers votre vidéo de test personnalisée.</p>
<p>Exécutez test.py pour initier la détection de somnolence en temps réel en utilisant le modèle pré-entraîné.</p>
<pre><code>python test.py</code></pre>
<p>Le script affichera le flux vidéo avec les événements anormaux détectés mis en évidence.</p>

<h3>Dépendances</h3>
<p>Assurez-vous d'avoir installé les bibliothèques requises :</p>
<pre><code>pip install opencv-python numpy keras imutils</code></pre>

<p>N'hésitez pas à explorer, expérimenter et contribuer pour améliorer l'efficacité de ce système de détection de somnolence. Bonne route en toute sécurité !</p>

</body>
</html>
