import pandas as pd
import numpy as np



def recode_decimal_millier(df):
    '''Recode les notes IMDB et Vote IMDB pour avoir respectivement un chiffre décimal et une notation sans séparateur de millier
    '''
    df['Note IMDB'] = [x.replace(',', '.') for x in df['Note IMDB']]
    df['Vote IMDB'] = [x.replace(' ', '') for x in df['Vote IMDB']]
    return df



def Recode_Certification(df):
    ''' Fonction qui permet de recoder les valeurs des Certifications selon le MPAA (Motion Picture Association of America). Les "See all certifications »" sont en réalité des valeurs manquante. "Unrated" est différent des valeurs manquantes. Ce sont des films qui ne sont pas noté volontairement
    '''
    return df.replace({"Tous Publics": "G", "Tous Public": "G",
                 "7":"G","10":"G","Tous publics":"G",
                 "Tous publics avec avertissement":"PG",
                 "13":"PG-13","12":"PG-13", "12 avec avertissement":"PG-13",
                 "16":"R",
                 "18":"NC-17",
                 "Not Rated": "Unrated",
                 "See all certifications »": None
                })



def supression_na(df):
    ''' Supression des NA pour le Box office mondial (variable qui nous interesse), Metascore car l'imputation de valeur n'aurais aucun sens tout comme le budget et la durée.'''
    return df.dropna(axis=0, subset=["Box Mondial", "Metascore",
                                         "Budget", "Duree"])



def remplace_na_USA(df):
    ''' 
    Les valeurs manquantes du Box office USA et recette Semaine 1 USA sont du au fait que le film n'est pas sortie dans les salles américaines. La fonction permet donc de remplacer les valeurs manquantes par 0
    '''
    return df.fillna(value={"Box USA":0,"Semaine 1":0})

def conversion_type(df):
    ''' Convertie tous les type des variables. Ceci est fait manuellement pour "Certification", "Langues", "Pays", "Note IMDB" et "Vote IMDB"
    '''
    df[["Langues","Pays","Certification"]]= df[["Langues","Pays","Certification"]].astype("category")
    df['Note IMDB'] = df['Note IMDB'].astype(float)
    df['Vote IMDB'] = df['Vote IMDB'].astype(int)
    return df.convert_dtypes()




def recode_genre(df):
    '''Fonction qui permet d'ajouter chaque genre en variable catégorielle individuelle  '''
    df['Genre'] = [x.replace(" ","").split('|') for x in df['Genre']]
    df['id']=range(0,len(df))
    for i in range(0,len(df)):
        for x in df.loc[df['id']==i,'Genre']:
            for y in x:
                df.loc[df['id']==i,y]=1
    df = df.fillna(value={"Action":0,
                            "Crime":0,
                            "Drama":0,
                            "Thriller":0,
                            "Adventure":0,
                            "Sci-Fi":0,
                            "Fantasy":0,
                            "Western":0,
                            "War":0,
                            "Mystery":0,
                            "Biography":0,
                            "Comedy":0,
                            "Animation":0,
                            "Family":0,
                            "Romance":0,
                            "Music":0,
                            "History":0,
                            "Sport":0,
                            "Horror":0,
                            "Musical":0,
                           })
    
    return df


    

def suppression_colonnes(df):
    '''Suppression des colonnes Titre, Producteurs, id et Genre qui ne sont pas utiles
    '''
    return df.drop(axis = 1,labels=["Titre", "Genre", "id", "Producteur"])
