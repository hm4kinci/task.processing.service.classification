from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence
from flair.trainers import ModelTrainer

train = SentenceDataset(
    [

        Sentence('email').add_label('contact_type', 'email'),
        Sentence('21 Jan: email client about signing them up for phrase 2 of Project Alpha').add_label('contact_type', 'email'),
        Sentence('Project Alpha: email client about signing them up for phase 2').add_label('contact_type', 'email'),
        Sentence('emailed client').add_label('contact_type', 'email'),
        Sentence('sent an email to the client about the project').add_label('contact_type', 'email'),
        Sentence('sent email to the client').add_label('contact_type', 'email'),
        Sentence('e-mailed to the client').add_label('contact_type', 'email'),
        Sentence('e-mailing to the client').add_label('contact_type', 'email'),
        Sentence('e-mail to the client').add_label('contact_type', 'email'),
        Sentence('sent email to the client about the new offer').add_label('contact_type', 'email'),
        Sentence('as I planned yesterday I emailed client').add_label('contact_type', 'email'),
        Sentence('emailing recent discussions').add_label('contact_type', 'email'),
        Sentence('today(project alpha) emailing recent discussions').add_label('contact_type', 'email'),
        Sentence('emailed client to schedule a meeting next week').add_label('contact_type', 'email'),
        Sentence('sent an email to client to schedule a meeting next week').add_label('contact_type', 'email'),
        Sentence('sent an email to client to set up a skype call').add_label('contact_type', 'email'),

        Sentence('21 Jan: call with client about signing them up for phase 2 of Project').add_label('contact_type', 'call'),
        Sentence('21 Jan: phone call with client about signing them up for phase 2 of Project').add_label('contact_type', 'call'),
        Sentence('21 Jan: skype with client about signing them up for phase 2 of Project').add_label('contact_type', 'call'),
        Sentence('Project Alpha: call with client about signing them up for phase 2').add_label('contact_type', 'call'),
        Sentence('Project Alpha: phone call with client about signing them up for phase 2').add_label('contact_type', 'call'),
        Sentence('Project Alpha: skype with client about signing them up for phase 2').add_label('contact_type', 'call'),
        Sentence('Phoned client about the phase 2 of the Project').add_label('contact_type', 'call'),
        Sentence('about the phase 2 of the Project, I called client').add_label('contact_type', 'call'),
        Sentence('phoning client about the project').add_label('contact_type', 'call'),
        Sentence('call with client').add_label('contact_type', 'call'),
        Sentence('calling').add_label('contact_type', 'call'),
        Sentence('skype call').add_label('contact_type', 'call'),
        Sentence('skype video call with Mary to discuss').add_label('contact_type', 'call'),
        Sentence('skyped client').add_label('contact_type', 'call'),
        Sentence('called client to inform them').add_label('contact_type', 'call'),
        Sentence('have a call about the project').add_label('contact_type', 'call'),
        Sentence('give a call').add_label('contact_type', 'call'),
        Sentence('telephone call with client about the project').add_label('contact_type', 'call'),
        Sentence('today: telephone call with client about the project').add_label('contact_type', 'call'),
        Sentence('18 December: telephone call with client about the project').add_label('contact_type', 'call'),
        Sentence('December 18th: telephone call with client about the project').add_label('contact_type', 'call'),
        Sentence('called client to schedule a meeting next week').add_label('contact_type', 'call'),
        Sentence('called client to set up a meeting next week').add_label('contact_type', 'call'),

        Sentence('21 Jan: meeting with client about signing them up for phrase 2 of Project Alpha').add_label('contact_type', 'meeting'),
        Sentence('21 Jan: meet with client about signing them up for phrase 2 of Project Alpha').add_label('contact_type', 'meeting'),
        Sentence('Project Alpha: meeting with client about signing them up for phase 2').add_label('contact_type', 'meeting'),
        Sentence('Project Alpha: meeting with client about signing them up for phase 2').add_label('contact_type', 'meeting'),
        Sentence('meet up with them').add_label('contact_type', 'meeting'),
        Sentence('meeting with client').add_label('contact_type', 'meeting'),
        Sentence('met client to discuss project').add_label('contact_type', 'meeting'),
        Sentence('meet with client at their office to review project').add_label('contact_type', 'meeting'),
        Sentence('meet').add_label('contact_type', 'meeting'),
        Sentence('set up a meeting').add_label('contact_type', 'meeting'),
        Sentence('joined a meeting').add_label('contact_type', 'meeting'),
        Sentence('participate in a meeting').add_label('contact_type', 'meeting'),
        Sentence('represent client at a meeting').add_label('contact_type', 'meeting'),
        Sentence('10 October: represent client at a meeting').add_label('contact_type', 'meeting'),
        Sentence('October 10th: represent client at a meeting').add_label('contact_type', 'meeting'),
        Sentence('met with client and decided to discuss it later over a call').add_label('contact_type', 'meeting'),
        Sentence('met with client and agreed to continue over email').add_label('contact_type', 'meeting'),

    ])

test = SentenceDataset(
    [
        Sentence('21 Jan: Phoned John S about signing them up for phrase 2 of Project Alpha.').add_label('contact_type',
                                                                                                         'call'),
        Sentence('Met with Jeremy today to discuss Project Beta').add_label('contact_type', 'meeting'),
        Sentence('Skype with Andrew at 2 pm today').add_label('contact_type', 'call'),
        Sentence('We had a 1 hour call with Denise').add_label('contact_type', 'call'),
        Sentence('Had a quick call to discuss the offer').add_label('contact_type', 'call'),
        Sentence('I was on skype with Paul all day').add_label('contact_type', 'call'),
        Sentence('I have set up a meeting tomorrow').add_label('contact_type', 'meeting'),
        Sentence('I emailed the latest report').add_label('contact_type', 'email'),
        Sentence('I sent an email to Mark').add_label('contact_type', 'email'),
        Sentence('I will email those files to you').add_label('contact_type', 'email'),
        Sentence('phoned Jeremy').add_label('contact_type', 'call'),
        Sentence('emailed Jeremy').add_label('contact_type', 'email'),
        Sentence('called Jeremy').add_label('contact_type', 'call'),
        Sentence('meet with Jeremy').add_label('contact_type', 'meeting'),
        Sentence('meeting w Deborah').add_label('contact_type', 'meeting'),
        Sentence('the client scheduled a meeting at their office').add_label('contact_type', 'meeting'),
        Sentence('had a meeting with client and agreed to continue over email').add_label('contact_type', 'meeting'),
        Sentence('had a call with client and to arrange a meeting tomorrow').add_label('contact_type', 'call'),
    ])

corpus = Corpus(train=train, test=test)

tars = TARSClassifier.load('tars-base')

tars.add_and_switch_to_new_task("contact_type", label_dictionary=corpus.make_label_dictionary())

trainer = ModelTrainer(tars, corpus)

trainer.train(base_path='../pretrained/contact_type_model/',
              learning_rate=0.02,
              mini_batch_size=1,
              max_epochs=15,
              train_with_dev=True,
              )