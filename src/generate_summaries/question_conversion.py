# https://stackoverflow.com/questions/55243829/easy-way-for-sentence-to-question-using-spacy-in-python
from textacy.spacier import utils
import spacy

nlp = spacy.load("en_core_web_sm")
inp = (
    input()
)  # The ingredients for an omelette are eggs, bacon, cheese, and onions, I do zip your code
doc = nlp(
    inp
)  # John will be finishing his homework   We were taking classes for decades
string, label = [], ""

for sentence in doc.sents:
    root = sentence.root
    for i in sentence.ents:
        if (
            len(utils.get_subjects_of_verb(root))
            or len(utils.get_objects_of_verb(root)) > 0
        ):
            label = i.label_
    print(root.tag_)
    print(root.lemma_)
    print(label)
    if len(utils.get_subjects_of_verb(root)) > 0:
        if root.lemma_ == "be":
            if label == "PERSON":
                ques = (
                    "Who "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            elif label == "QUANTITY":
                ques = (
                    "How "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            elif label == "MONEY":
                ques = (
                    "How much "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            elif label == "TIME":
                ques = (
                    "When "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            elif label == "GPE":
                ques = (
                    "Where "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            elif label == "PRODUCT":
                ques = (
                    "What "
                    + str(root)
                    + " "
                    + str(utils.get_subjects_of_verb(root)[0])
                    + " ?"
                )
            else:
                for to in doc:
                    if to.pos_ == "VERB":
                        q = to.i
                question = doc
                for r in doc[q].rights:
                    for x, y in enumerate(r.subtree):
                        question = str(question).replace(str(y), "")
                print(str(question) + " what?")
                break
        elif root.lemma_ == "do":
            if label in ["QUANTITY", "MONEY", "CARDINAL"] and utils.get_objects_of_verb(
                root
            )[0].dep_ in ["pobj", "dobj"]:
                for child in utils.get_objects_of_verb(root)[0].children:
                    if child.tag_ in ["CD", "#"]:
                        if root.tag_ == "VBD" or str(
                            utils.get_subjects_of_verb(root)[0]
                        ) in ["I", "You", "We", "They", "He", "She", "It"]:
                            ques = (
                                "How many did "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                            "I",
                            "You",
                            "We",
                            "They",
                        ]:
                            ques = (
                                "How many do "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif (
                            str(utils.get_subjects_of_verb(root)[0]).upper()
                            in ["He", "She", "It"]
                            or label == "PERSON"
                        ):
                            ques = (
                                "How many does "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
            elif label in ["QUANTITY", "MONEY", "CARDINAL"]:
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "How much did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "How much do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "How much does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "TIME":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "When did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "When do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "When does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "GPE":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "Where did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "Where do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "Where does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            else:
                if root.tag_ in ["VBP", "VB", "VBG", "VBZ"] or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They"]:
                    ques = (
                        "What do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    root.tag_ in ["VBP", "VB", "VBG", "VBZ"]
                    or str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "What does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    root.tag_ == "VBD"
                    or str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["I", "You", "We", "They", "He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "What did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
        elif root.lemma_ == "have":
            print(root.lemma_)
            if label in ["QUANTITY", "MONEY", "CARDINAL"] and utils.get_objects_of_verb(
                root
            )[0].dep_ in ["pobj", "dobj"]:
                for child in utils.get_objects_of_verb(root)[0].children:
                    if child.tag_ in ["CD", "#"]:
                        if root.tag_ == "VBD" or str(
                            utils.get_subjects_of_verb(root)[0]
                        ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                            ques = (
                                "How many did "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                            "I",
                            "You",
                            "We",
                            "They",
                        ]:
                            ques = (
                                "How many do "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif (
                            str(utils.get_subjects_of_verb(root)[0]).upper()
                            in ["He", "She", "It"]
                            or label == "PERSON"
                        ):
                            ques = (
                                "How many does "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
            elif label in ["QUANTITY", "MONEY", "CARDINAL"]:
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "How much did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "How much do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "How much does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "TIME":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "When did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "When do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "When does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "GPE":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "Where did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "Where do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "Where does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            else:
                if root.tag_ in ["VBP", "VB", "VBG", "VBZ"] and str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They"]:
                    ques = "Do " + str(sentence).replace(root.text, root.lemma_) + "?"
                elif (
                    root.tag_ in ["VBP", "VB", "VBG", "VBZ"]
                    or str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = "Does " + str(sentence).replace(root.text, root.lemma_) + "?"
                elif (
                    root.tag_ == "VBD"
                    or str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["I", "You", "We", "They", "He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = "Did " + str(sentence).replace(root.text, root.lemma_) + "?"
        elif root.lemma_ == "need":
            if label in ["QUANTITY", "MONEY", "CARDINAL"] and utils.get_objects_of_verb(
                root
            )[0].dep_ in ["pobj", "dobj"]:
                for child in utils.get_objects_of_verb(root)[0].children:
                    if child.tag_ in ["CD", "#"]:
                        if root.tag_ == "VBD" or str(
                            utils.get_subjects_of_verb(root)[0]
                        ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                            ques = (
                                "How many did "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                            "I",
                            "You",
                            "We",
                            "They",
                        ]:
                            ques = (
                                "How many do "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
                        elif (
                            str(utils.get_subjects_of_verb(root)[0]).upper()
                            in ["He", "She", "It"]
                            or label == "PERSON"
                        ):
                            ques = (
                                "How many does "
                                + str(utils.get_subjects_of_verb(root)[0])
                                + " "
                                + root.lemma_
                                + "?"
                            )
            elif label in ["QUANTITY", "MONEY", "CARDINAL"]:
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "How much did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "How much do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "How much does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "TIME":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "When did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "When do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "When does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            elif label == "GPE":
                if root.tag_ == "VBD" or str(
                    utils.get_subjects_of_verb(root)[0]
                ).upper() in ["I", "You", "We", "They", "He", "She", "It"]:
                    ques = (
                        "Where did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "Where do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "Where does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
            else:
                if str(utils.get_subjects_of_verb(root)[0]).upper() in [
                    "I",
                    "You",
                    "We",
                    "They",
                ]:
                    ques = (
                        "What do "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "What does "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )
                elif (
                    root.tag_ == "VBD"
                    or str(utils.get_subjects_of_verb(root)[0]).upper()
                    in ["I", "You", "We", "They", "He", "She", "It"]
                    or label == "PERSON"
                ):
                    ques = (
                        "What did "
                        + str(utils.get_subjects_of_verb(root)[0])
                        + " "
                        + root.lemma_
                        + "?"
                    )

        string.append(ques)

print(string)
