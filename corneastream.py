import streamlit as st

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from collections import Counter
import s3fs













def write_navigation_bar():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Go to", ["Cornea rare disease finder", "Study", "About"])
    if page == "Cornea rare disease finder":
        write_main_page()
    elif page == "Study":
        write_page_1()
    elif page == "About":
        write_page_2()



def write_main_page():
    
    fs = s3fs.S3FileSystem(anon=False)
    filename = "sebbax/example.txt"


    with fs.open(filename) as f:
          content = f.read()
          f.write('neue Zeile')  

   
    
    st.header(content)

    yes_no_unknown = {"I'm not sure!": 'unknown', 'Yes': 'Yes', 'No': 'No'}

    myvariabledict = {"I'm not sure.": "unknown", "0 -10 years" :"0", "11 - 20 years": "1", "21 - 30 years": "2", "31 years and older": "3"}

    myvariabledict6 = {"Unknown (family history unclear/ only affected family member)": 'unknown', 'AD': 'AD', 'AR': 'AR', 'No inheritance': 'NO', 'X-linked': 'X'}

    myvariabledict7 = {"I'm not sure!": 'unknown', 'Unilateral.': 'Yes', 'Bilateral.': 'No'}

    myvariabledict3 = {"I'm not sure!": 'unknown', 'Endothelium': 'Endothelium', 'Epithelium': 'Epithelium', 'Stroma': 'Stroma', 'Stroma/Endothelium': 'Endo/Stroma'}


    #### Fragebogen #####

    myvariables = st.selectbox("Age of first time clinical appearance?",  list(myvariabledict.keys()))
    myvariable = myvariabledict[myvariables]

    myvariables2 = st.selectbox("Recurrent erosions?", list(yes_no_unknown.keys()))
    myvariable2 = yes_no_unknown[myvariables2]


    myvariables5 = st.selectbox("Progressive?", list(yes_no_unknown.keys()))
    myvariable5 = yes_no_unknown[myvariables5]



    myvariables6 = st.selectbox("Suspected Inheritance?", list(myvariabledict6.keys()))
    myvariable6 = myvariabledict6[myvariables6]



    myvariables7 = st.selectbox("Laterality?", list(myvariabledict7.keys()))
    myvariable7 = myvariabledict7[myvariables7]



    myvariables3 = st.selectbox("Primarily affected layer?", list(myvariabledict3.keys()))
    myvariable3 = myvariabledict3[myvariables3]




    myvariables4 = st.selectbox("Corneal thinning?", list(yes_no_unknown.keys()))
    myvariable4 = yes_no_unknown[myvariables4]


    myvariables22= st.selectbox("Corneal steepening?", list(yes_no_unknown.keys()))
    myvariable22 = yes_no_unknown[myvariables22]


    
    
    myvariables8 = st.selectbox("Epithelial microcysts?", list(yes_no_unknown.keys()))
    myvariable8 = yes_no_unknown[myvariables8]

    
    
    myvariables9 = st.selectbox("Epithelial thickening?", list(yes_no_unknown.keys()))
    myvariable9 = yes_no_unknown[myvariables9]

    
    myvariables14= st.selectbox("Epithelial or stromal honeycomb?", list(yes_no_unknown.keys()))
    myvariable14 = yes_no_unknown[myvariables14]


    myvariables15= st.selectbox("Epithelial or stromal geographical deposits?", list(yes_no_unknown.keys()))
    myvariable15 = yes_no_unknown[myvariables15]


    
    myvariables10 = st.selectbox("Stromal Rings, Stars or breadcrumbs?", list(yes_no_unknown.keys()))
    myvariable10 = yes_no_unknown[myvariables10]
    
    myvariables11 = st.selectbox("Stromal snowflakes?", list(yes_no_unknown.keys()))
    myvariable11 = yes_no_unknown[myvariables11]
    
    myvariables12 = st.selectbox("Stromal cloudy appearance?", list(yes_no_unknown.keys()))
    myvariable12 = yes_no_unknown[myvariables12]

    myvariables13 = st.selectbox("Stromal peripheral arcus?", list(yes_no_unknown.keys()))
    myvariable13 = yes_no_unknown[myvariables13]
    
    myvariables16 = st.selectbox("Stroma predescemetal haze?", list(yes_no_unknown.keys()))
    myvariable16 = yes_no_unknown[myvariables16]

    myvariables17= st.selectbox("Stromal crystals?", list(yes_no_unknown.keys()))
    myvariable17 = yes_no_unknown[myvariables17]
    
    myvariables18= st.selectbox("Diffuse stromal haze?", list(yes_no_unknown.keys()))
    myvariable18 = yes_no_unknown[myvariables18]


    myvariables19= st.selectbox("Deep stromal diffuse deposits?", list(yes_no_unknown.keys()))
    myvariable19 = yes_no_unknown[myvariables19]
    
    myvariables20 = st.selectbox("Irregular posterior surface?", list(yes_no_unknown.keys()))
    myvariable20 = yes_no_unknown[myvariables20]
    
    myvariables21= st.selectbox("Beaten metal appearance?", list(yes_no_unknown.keys()))
    myvariable21 = yes_no_unknown[myvariables21]

    myvariables23= st.selectbox("Tiny dots on the posterior corneal surface?", list(yes_no_unknown.keys()))
    myvariable23 = yes_no_unknown[myvariables23]

    ###Fragebogen ende ####
    
    ergebnisliste = [myvariable, myvariable2 ,myvariable3 ,myvariable4, myvariable5, myvariable6, myvariable7,myvariable8,myvariable9,myvariable10,myvariable11,
                        myvariable12,myvariable13,myvariable14,myvariable15, myvariable16,myvariable17, myvariable18, myvariable19, myvariable20, myvariable21, myvariable22, myvariable23 ]

    liste = transform_ergebnisse(ergebnisliste)

    neue_liste = sum (liste, [])

    model, clf, neigh, log, clf, mlp, clf_svm = train()




    ergebnisse = []

    tree = str(model.predict([neue_liste])).strip('[]')

    forest = str(clf.predict([neue_liste])).strip('[]')

    neighbors = str(neigh.predict([neue_liste])).strip('[]')

    regression = str(log.predict([neue_liste])).strip('[]')

    bayesian = str(clf.predict([neue_liste])).strip('[]')

    perception = str(mlp.predict([neue_liste])).strip('[]')

    support = str(clf_svm.predict([neue_liste])).strip('[]')

    ergebnisse = [tree, forest, regression, bayesian, perception, support]

    ergebnisse = list(ergebnisse)
    c = Counter(ergebnisse)
    haeufigkeit = c.most_common()
    print(len(haeufigkeit))



    first_place = haeufigkeit[0]
    try:
        second_place = haeufigkeit[1]
        second_place = second_place[0]
        second_place = second_place.strip('[]').replace("'", "")
    except:
        second_place = 'Die oben genannte ist die einzige Lösung.'

    first_place = first_place[0]
    first_place = first_place.strip('[]')


    erster = first_place
    first_place = first_place.replace("'", "")

    with open('read.txt', 'w') as file:
            file.writelines(f"{first_place}")
                    

    

    
    if myvariable and myvariable2:
        st.success("The most likely disease given your input is : {} and {}".format(first_place, second_place))


def train (): 
        #Lade csv
    daten = pd.read_csv('corneal_dystrophies - corneal_dystrophies _data Kopie(2).csv')


    daten = daten.fillna('unknown')
    daten = daten.replace('y', 'yes')
    daten = daten.replace('n', 'no')
        #daten.head(5)


    features = list(daten.head(0))
    features = features[4::]


        #Modifiziere Eingabe

    for i in features:
        daten[i] = daten[i].replace('yes', i)
        daten[i] = daten[i].replace('no', f'not_{i}')
        daten[i] = daten[i].replace('unknown', f'unknown_{i}')


        # One-Hot encoding aller features

    daten_encoded = pd.get_dummies(data=daten, columns=['decade of diagnosis','recurrent erosions', 'primarily affected layer','corneal thinning', 'non progressive','inheritance', 'may be unilateral', 'microcysts', 'epithelial thickening',
                                        'stroma: rings / stars', 'stroma: central snowflakes / lines', 'stroma: cloudy appearance', 'stroma: arcus', 'stroma: honeycomb', 'stroma: confluent geographic', 'stroma: pre decemetal haze', 'stromal crystals',
                                        'diffuse stromal haze ', 'deep stromal diffuse deposits', 'irregular posterior corneal surface', 'beaten metal appearance of corneal surface', 'corneal steepening', 'tiny dots on the posterior corneal surface'])
    #daten_encoded

    daten_encoded.to_csv('daten_encoded.csv')

    # ergebnisvektor
    y = daten["Name"]
    target_array = y.values
    #target_array


    namen = list(daten_encoded.head(0))

    namen = namen[4::]



    X = daten_encoded[namen].values
    y = target_array

        #Decision Tree
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

    model = DecisionTreeClassifier(random_state = 1)
    model.fit(X, y)


        #Random Forest
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

        #k nearest Neighbours

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

        #Logistic Rergression

    log = LogisticRegression()
    log.fit(X,y)

        #categorial bayes
    clf = CategoricalNB()
    clf.fit(X, y)

        #multi layer perceptron
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    mlp.fit(X,y)

        #support vector machines
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(X,y)


    return (model, clf, neigh, log, clf, mlp, clf_svm)        

      

    

def write_page_2():
    st.header("About")
    st.write("This is page 2.")


def transform_ergebnisse(liste):
            prediction_list = []
        #decade of diagnosis
            if liste[0] == "0":
                prediction_list.append([1,0,0,0,0])
            elif liste[0] == '1':
                prediction_list.append([0,1,0,0,0])
            elif liste[0] == '2':
                prediction_list.append([0,0,1,0,0])
            elif liste[0] == '3':
                prediction_list.append([0,0,0,1,0])
            elif liste[0] == 'unknown':
                prediction_list.append([0,0,0,0,1])

            #frage_2 = input('keine Wiederkehrenden erosionen, wiederkehrende erosionen, unbekannt')
            if liste[1] == "No":
                prediction_list.append([1,0,0])
            elif liste[1] == 'Yes':
                prediction_list.append([0,1,0])
            elif liste[1] == 'unknown':
                prediction_list.append([0,0,1])

            #frage_3 = input('endo, epi, stro, stro, endo, unbekannt')
            if liste[2] == "Endothelium":
                prediction_list.append([1,0,0,0,0])
            elif liste[2] == 'Epithelium':
                prediction_list.append([0,1,0,0,0])
            elif liste[2] == 'Stroma':
                prediction_list.append([0,0,1,0,0])
            elif liste[2] == 'Stroma/Endo':
                prediction_list.append([0,0,0,1,0])
            elif liste[2] == 'unknown':
                prediction_list.append([0,0,0,0,1])

            #frage_4 = input('keine Korneaausdünnung, unbekannt')

            if liste[3] == "Yes":
                prediction_list.append([1,0])
            elif liste[3] == "No":
                prediction_list.append([0,1])
            elif liste[3] == 'unknown':
                prediction_list.append([0,0])

            #frage_5 = input('nicht progressiv, unbekannt ')
            if liste[4] == "No":
                prediction_list.append([1,0])
            elif liste[4] == "Yes":
                prediction_list.append([0,1])
            elif liste[4] == 'unknown':
                prediction_list.append([0,0])

            #frage_6 = input('Vererbung:  AD,  AR,  NO,  X,  XD, XR')
            if liste[5] == "AD":
                prediction_list.append([1,0,0,0])
            elif liste[5] == 'AR':
                prediction_list.append([0,1,0,0])
            elif liste[5] == 'NO':
                prediction_list.append([0,0,1,0])
            elif liste[5] == 'X':
                prediction_list.append([0,0,0,1])
            elif liste[5] == 'unknown':
                prediction_list.append([0,0,0,0])

            #frage_7 = input('unilateral, unbekannt')
            if liste[6] == "Yes":
                prediction_list.append([1,0])
            elif liste[6] == 'No':
                prediction_list.append([0,1])
            elif liste[6] == 'unknown':
                prediction_list.append([0,0])

            #frage_8 = input('mikrozysten, unbekannt')
            if liste[7] == "Yes":
                prediction_list.append([1,0])
            elif liste[7] == 'No':
                prediction_list.append([0,1])
            elif liste[7] == 'unknown':
                prediction_list.append([0,0])

            #frage_9 = input('Epithelverdickung, unbekannt')
            if liste[8] == "Yes":
                prediction_list.append([1,0])
            elif liste[8] == 'No':
                prediction_list.append([0,1])
            elif liste[8] == 'unknown':
                prediction_list.append([0,0])

            #frage_95 = input('stroma: rings / stars   , unbekannt ')
            if liste[9] == "Yes":
                prediction_list.append([0,1])
            elif liste[9] == 'No':
                prediction_list.append([1,0])
            elif liste[9] == 'unknown':
                prediction_list.append([0,0])


            #frage_10 = input('stroma: central snowflakes,  unbekannt')
            if liste[10] == "Yes":
                prediction_list.append([0,1])
            elif liste[10] == 'No':
                prediction_list.append([1,0])
            elif liste[10] == 'unknown':
                prediction_list.append([0,0])

            #frage_11 = input('stroma: cloudy appearance, unbekannt')
            if liste[11] == "Yes":
                prediction_list.append([0,1])
            elif liste[11] == 'No':
                prediction_list.append([1,0])
            elif liste[11] == 'unknown':
                prediction_list.append([0,0])


            #frage_12 = input('stroma: arcus_stroma, unbekannt ')
            if liste[12] == "Yes":
                prediction_list.append([0,1])
            elif liste[12] == 'No':
                prediction_list.append([1,0])
            elif liste[12] == 'unknown':
                prediction_list.append([0,0])

            #frage_13 = input('stroma: honeycomb, unbekannt')
            if liste[13] == "Yes":
                prediction_list.append([0,1])
            elif liste[13] == 'No':
                prediction_list.append([1,0])
            elif liste[13] == 'unknown':
                prediction_list.append([0,0])

            #frage_14 = input('stroma: confluent geographic, unbekannt')
            if liste[14] == "Yes":
                prediction_list.append([0,1])
            elif liste[14] == 'No':
                prediction_list.append([1,0])
            elif liste[14] == 'unknown':
                prediction_list.append([0,0])

            #frage_15 = input('stroma: pre decemetal haze, unbekannnt ')
            if liste[15] == "Yes":
                prediction_list.append([0,1])
            elif liste[15] == 'No':
                prediction_list.append([1,0])
            elif liste[15] == 'unknown':
                prediction_list.append([0,0])

            #frage_16 = stromal crystals
            if liste[16] == "No":
                prediction_list.append([0,1])
            elif liste[16] == 'Yes':
                prediction_list.append([1,0])
            elif liste[16] == 'unknown':
                prediction_list.append([0,0])

            #frage_17 = diffuse stromal haze
            if liste[17] == "No":
                prediction_list.append([0,1])
            elif liste[17] == 'Yes':
                prediction_list.append([1,0])
            elif liste[17] == 'unknown':
                prediction_list.append([0,0])

            #frage_18 = deep stromal dffuse deposits
            if liste[18] == "No":
                prediction_list.append([0,1])
            elif liste[18] == 'Yes':
                prediction_list.append([1,0])
            elif liste[18] == 'unknown':
                prediction_list.append([0,0])

            #frage_19 = irregular posterior surface
            if liste[19] == "No":
                prediction_list.append([0,1])
            elif liste[19] == 'Yes':
                prediction_list.append([1,0])
            elif liste[19] == 'unknown':
                prediction_list.append([0,0])

            #frage_20 = beaten metal appearance
            if liste[20] == "No":
                prediction_list.append([0,1])
            elif liste[20] == 'Yes':
                prediction_list.append([1,0])
            elif liste[20] == 'unknown':
                prediction_list.append([0,0])

            #frage_21 = corneael steepening
            if liste[21] == "No":
                prediction_list.append([0,1])
            elif liste[21] == 'Yes':
                prediction_list.append([1,0])
            elif liste[21] == 'unknown':
                prediction_list.append([0,0])

            #frage_22 = tiny dots on the posterior surface
            if liste[22] == "No":
                prediction_list.append([0,1])
            elif liste[22] == 'Yes':
                prediction_list.append([1,0])
            elif liste[22] == 'unknown':
                prediction_list.append([0,0])

            return prediction_list    



def write_page_1():
    yes_no_unknown = {"I'm not sure!": 'unknown', 'Yes': 'Yes', 'No': 'No'}

    myvariabledict = {"I'm not sure.": "unknown", "0 -10 years" :"0", "11 - 20 years": "1", "21 - 30 years": "2", "31 years and older": "3"}

    myvariabledict6 = {"Unknown (family history unclear/ only affected family member)": 'unknown', 'AD': 'AD', 'AR': 'AR', 'No inheritance': 'NO', 'X-linked': 'X'}

    myvariabledict7 = {"I'm not sure!": 'unknown', 'Unilateral.': 'Yes', 'Bilateral.': 'No'}

    myvariabledict3 = {"I'm not sure!": 'unknown', 'Endothelium': 'Endothelium', 'Epithelium': 'Epithelium', 'Stroma': 'Stroma', 'Stroma/Endothelium': 'Endo/Stroma'}

    myvariabledictwho = {"Corneal Expert": "expert", "Assistant doctor": "assistant", "Ophtalmologist": "ophtalmologist"}

    st.header("Accordion Example")

    # Use st.beta_expander to create an accordion section
    with st.expander("Questions about me:"):
       
        
        questions1 = st.selectbox("I am", list(myvariabledictwho.keys()))
        question1 = myvariabledictwho[questions1]
        

    
    with st.expander('This is the questionaire:'):

        #### Fragebogen #####

        myvariables = st.selectbox("Age of first time clinical appearance?",  list(myvariabledict.keys()))
        myvariable = myvariabledict[myvariables]

        myvariables2 = st.selectbox("Recurrent erosions?", list(yes_no_unknown.keys()))
        myvariable2 = yes_no_unknown[myvariables2]


        myvariables5 = st.selectbox("Progressive?", list(yes_no_unknown.keys()))
        myvariable5 = yes_no_unknown[myvariables5]



        myvariables6 = st.selectbox("Suspected Inheritance?", list(myvariabledict6.keys()))
        myvariable6 = myvariabledict6[myvariables6]



        myvariables7 = st.selectbox("Laterality?", list(myvariabledict7.keys()))
        myvariable7 = myvariabledict7[myvariables7]



        myvariables3 = st.selectbox("Primarily affected layer?", list(myvariabledict3.keys()))
        myvariable3 = myvariabledict3[myvariables3]




        myvariables4 = st.selectbox("Corneal thinning?", list(yes_no_unknown.keys()))
        myvariable4 = yes_no_unknown[myvariables4]


        myvariables22= st.selectbox("Corneal steepening?", list(yes_no_unknown.keys()))
        myvariable22 = yes_no_unknown[myvariables22]


        
        
        myvariables8 = st.selectbox("Epithelial microcysts?", list(yes_no_unknown.keys()))
        myvariable8 = yes_no_unknown[myvariables8]

        
        
        myvariables9 = st.selectbox("Epithelial thickening?", list(yes_no_unknown.keys()))
        myvariable9 = yes_no_unknown[myvariables9]

        
        myvariables14= st.selectbox("Epithelial or stromal honeycomb?", list(yes_no_unknown.keys()))
        myvariable14 = yes_no_unknown[myvariables14]


        myvariables15= st.selectbox("Epithelial or stromal geographical deposits?", list(yes_no_unknown.keys()))
        myvariable15 = yes_no_unknown[myvariables15]


        
        myvariables10 = st.selectbox("Stromal Rings, Stars or breadcrumbs?", list(yes_no_unknown.keys()))
        myvariable10 = yes_no_unknown[myvariables10]
        
        myvariables11 = st.selectbox("Stromal snowflakes?", list(yes_no_unknown.keys()))
        myvariable11 = yes_no_unknown[myvariables11]
        
        myvariables12 = st.selectbox("Stromal cloudy appearance?", list(yes_no_unknown.keys()))
        myvariable12 = yes_no_unknown[myvariables12]

        myvariables13 = st.selectbox("Stromal peripheral arcus?", list(yes_no_unknown.keys()))
        myvariable13 = yes_no_unknown[myvariables13]
        
        myvariables16 = st.selectbox("Stroma predescemetal haze?", list(yes_no_unknown.keys()))
        myvariable16 = yes_no_unknown[myvariables16]

        myvariables17= st.selectbox("Stromal crystals?", list(yes_no_unknown.keys()))
        myvariable17 = yes_no_unknown[myvariables17]
        
        myvariables18= st.selectbox("Diffuse stromal haze?", list(yes_no_unknown.keys()))
        myvariable18 = yes_no_unknown[myvariables18]


        myvariables19= st.selectbox("Deep stromal diffuse deposits?", list(yes_no_unknown.keys()))
        myvariable19 = yes_no_unknown[myvariables19]
        
        myvariables20 = st.selectbox("Irregular posterior surface?", list(yes_no_unknown.keys()))
        myvariable20 = yes_no_unknown[myvariables20]
        
        myvariables21= st.selectbox("Beaten metal appearance?", list(yes_no_unknown.keys()))
        myvariable21 = yes_no_unknown[myvariables21]

        myvariables23= st.selectbox("Tiny dots on the posterior corneal surface?", list(yes_no_unknown.keys()))
        myvariable23 = yes_no_unknown[myvariables23]

        ###Fragebogen ende ####
        
        ergebnisliste = [myvariable, myvariable2 ,myvariable3 ,myvariable4, myvariable5, myvariable6, myvariable7,myvariable8,myvariable9,myvariable10,myvariable11,
                            myvariable12,myvariable13,myvariable14,myvariable15, myvariable16,myvariable17, myvariable18, myvariable19, myvariable20, myvariable21, myvariable22, myvariable23 ]

        liste = transform_ergebnisse(ergebnisliste)

        neue_liste = sum (liste, [])

        model, clf, neigh, log, clf, mlp, clf_svm = train()




        ergebnisse = []

        tree = str(model.predict([neue_liste])).strip('[]')

        forest = str(clf.predict([neue_liste])).strip('[]')

        neighbors = str(neigh.predict([neue_liste])).strip('[]')

        regression = str(log.predict([neue_liste])).strip('[]')

        bayesian = str(clf.predict([neue_liste])).strip('[]')

        perception = str(mlp.predict([neue_liste])).strip('[]')

        support = str(clf_svm.predict([neue_liste])).strip('[]')

        ergebnisse = [tree, forest, regression, bayesian, perception, support]

        ergebnisse = list(ergebnisse)
        c = Counter(ergebnisse)
        haeufigkeit = c.most_common()
        print(len(haeufigkeit))



        first_place = haeufigkeit[0]
        try:
            second_place = haeufigkeit[1]
            second_place = second_place[0]
            second_place = second_place.strip('[]').replace("'", "")
        except:
            second_place = 'Die oben genannte ist die einzige Lösung.'

        first_place = first_place[0]
        first_place = first_place.strip('[]')


        erster = first_place
        first_place = first_place.replace("'", "")


                    

    

    
    if myvariable and myvariable2:
        st.success("The most likely disease given your input is : {} and {}".format(first_place, second_place))

 
                 

if __name__ == '__main__':
    write_navigation_bar()





