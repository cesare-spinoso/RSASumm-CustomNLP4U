import regex

# squality
question1 = """
Here are the 10 questions generated from the document:\n\nWhat is the nickname of Starrett Blade, the protagonist of the story?\nWho is Devil Garrett, and what is his reputation in the galaxy?\nWhat is the significance of the Barden Beam in the story?\nWhy is Starrett Blade searching for Devil Garrett, and what is his motivation?\nWhat is the nature of the relationship between Starrett Blade and the dark-haired girl, Miss Hinton?\nWhat is the purpose of the transmission set in the story, and how does it affect the plot?\nHow does Starrett Blade manage to escape from his prison cell, and what role does the girl play in his escape?\nWhat is the significance of the fork that Starrett Blade uses to disable the transmission set?\nHow does Starrett Blade's perception of the girl change throughout the story, and what does this reveal about his character?\nWhat is the ultimate fate of Devil Garrett, and how does this impact the story's conclusion?
"""
# qmsum
question2 = """
\n\nWhat is the main point of contention between Mr. Yves-Franois Blanchet and the Prime Minister, according to the document? Why did the Prime Minister receive a minority mandate from Quebeckers and Canadians, according to Mr. Yves-Franois Blanchet? What is the significance of the House of Commons giving its consent to extend the mandate of the Special Committee on the COVID-19 Pandemic, according to Right Hon. Justin Trudeau? Who will pay the price if the Prime Minister does not resolve the problem, according to Mr. Yves-Franois Blanchet? What is the purpose of the $14billion agreement proposed by the federal government, according to Right Hon. Justin Trudeau? When did the Prime Minister receive a minority mandate from Quebeckers and Canadians, according to Mr. Yves-Franois Blanchet? What is the difference between a prime minister with a majority and a monarch by divine right, according to Mr. Yves-Franois Blanchet? Why did the government want to buy the right to interfere in provincial and Quebec jurisdictions for $14billion, according to Mr. Yves-Franois Blanchet? What is the role of the federal government in ensuring the safety of Canadians, according to Right Hon. Justin Trudeau? Where was the Prime Minister on October 21, 2019, according to Mr. Yves-Franois Blanchet?
"""
matches = regex.findall(r"what[^\?]*\?|who[^\?]*\?|where[^\?]*\?|when[^\?]*\?|how[^\?]*\?|why[^\?]*\?", question1, flags=regex.IGNORECASE)
print(matches)
print(len(matches))

matches = regex.findall(r"what[^\?]*\?|who[^\?]*\?|where[^\?]*\?|when[^\?]*\?|how[^\?]*\?|why[^\?]*\?", question2, flags=regex.IGNORECASE)
print(matches)
print(len(matches))