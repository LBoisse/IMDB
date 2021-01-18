from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, Normalizer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def initialisation_pour_le_traitement(donnees_nettoyees, liste_variables_qualitatives_expl, liste_variables_quantitatives_expl, variable_interet):
    '''Fonction qui permet de préparer les données pour le ML. Permet le traitement des variables séléctionées. Créer les données d'entrainement et test. La fonction renvoie les échantillons entrainement et test '''
    ct = ColumnTransformer([
    (
        "Gestion quali", 
        Pipeline([
            ("GestionNaN", SimpleImputer(strategy="constant", fill_value="Donnee Manquante")),
            ("Numérisation", OneHotEncoder(sparse=False))
    
        ]), 
        liste_variables_qualitatives_expl
    ),
    (
        "Colonnes déja numériques",
        MinMaxScaler(),
        liste_variables_quantitatives_expl
    )
])
    X = ct.fit_transform(donnees_nettoyees)
    y = donnees_nettoyees[variable_interet].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_tr, X_te, y_tr, y_te




def Modeles_note_Metascore():
    '''Fonction créant les modèles pour la prediction des notes'''
    liste = list()
    for val_alpha in (10 ** n for n in range(-3, 3)):
        liste.append(Ridge(alpha=val_alpha))
                      
    for val_epsilon in (10 ** n for n in range(-3, 1)):
        for val_C in (10 ** n for n in range(-2, 5)):
            liste.append(SVR(epsilon=val_epsilon, C=val_C))
            
    for nb_estimateurs in (50, 100, 150, 200, 250, 300,350, 400):  
        liste.append(RandomForestRegressor(n_estimators=nb_estimateurs))
        
    for nb_voisins in range(3, 10):
        liste.append(KNeighborsRegressor(n_neighbors=nb_voisins))
    return liste


def Modeles_box_office():
    '''Fonction créant les modèles pour la prediction du box office'''
    liste = list()
                      
    for val_alpha in (10 ** n for n in range(-3, 3)):
        liste.append(Ridge(alpha=val_alpha))
                      
    for val_epsilon in (10 ** n for n in range(-3, 1)):
        for val_C in (10 ** n for n in range(7, 12)):
            liste.append(SVR(epsilon=val_epsilon, C=val_C))
            
    for nb_estimateurs in (50, 100, 150, 200, 250, 300,350, 400):  
        liste.append(RandomForestRegressor(n_estimators=nb_estimateurs))
        
    for nb_voisins in range(3, 10):
        liste.append(KNeighborsRegressor(n_neighbors=nb_voisins))
    return liste



def affichage_meilleurs_modeles_entrainement(liste, X_tr, y_tr):
    '''Fonction qui permet d'afficher les résultats des modèles séléctionnés'''
    resultats = dict()
    for modele in liste:
        resultats[modele] = cross_val_score(modele, X_tr, y_tr, cv=8)
        
    resultats_pour_tri = sorted([(scores.mean(), scores.std(), repr(modele)) for modele, scores in resultats.items()], reverse=True)
    
    for moyenne, ecart_type, nom_modele in resultats_pour_tri:
        print(f"{nom_modele:70} {moyenne:4.3}   {ecart_type:6.5}")
        
        

def score_modele_data_test(modele, X_tr, y_tr, X_te, y_te):
    '''Fonction qui permet d'afficher les score du modele choisis pour les donées d'entrainement et pour les données test '''
    modele.fit(X_tr, y_tr)
    score_train = cross_val_score(modele, X_tr, y_tr, cv=8)
    score_test = modele.score(X_te, y_te)
    print("Score données entrainement :", round(score_train.mean(),3))
    print("Score données test :",round(score_test,3))
    




    
    