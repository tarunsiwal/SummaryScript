import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')

def read_article(text):
    # Tokenize the article into sentences
    sentences = sent_tokenize(text)
    # Tokenize each sentence into words
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    return words

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    # Find the intersection of words
    all_words = list(set(sent1 + sent2))
    
    # Create vectors for each sentence
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build the vector representation for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    # Build the vector representation for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)
    
    return similarity_matrix

def generate_summary(text, top_n=10):
    # Tokenize the article into words and sentences
    sentences = read_article(text)
    
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    
    # Build the similarity matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Rank sentences using PageRank algorithm via networkx
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Sort the sentence scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Get the top sentences as the summary
    num_sentences = min(top_n, len(ranked_sentences))  # Limit to available sentences
    summary = ""
    for i in range(num_sentences):
        summary += " ".join(ranked_sentences[i][1]) + "\n"
    
    return summary


if __name__ == "__main__":
    article_text = """
   Was It People or Was It Aliens?
STORYTELLER: DOUG AVERILL, RETIRED OWNER AND MANAGER OF THE FLATHEAD LAKE LODGE
Doug Averill grew up as one of eight boys on his parents’ sprawling dude ranch, the Flathead Lake Lodge, in rural Montana. As a teen, the Averill boys ran wild. “We rode around as a little gang of cowboys,” he remembers. They’d saddle up and head off to check cattle on the three giant tracts of land the family managed, which formed a triangle around some of the state’s most remote rangelands.

One summer in the 1960s, the brothers came across a ghastly sight. There, on the ground, were three dead cows neatly arranged in a circle. No obvious wounds were visible, but their reproductive organs had been removed. “But there was never any blood. It was almost surgical removal,” Averill remembers.

During this decade, America was obsessed with aliens, and write-ups in the local newspapers posited that perhaps this was the work of extraterrestrials. People mused that aliens had taken the reproductive organs for testing. But one day, Averill and his friends came across a lance in their path. Attached to it was a cryptic note with a threatening message. “That’s when we thought, It’s gotta be people doing this,” he says.

Then things got really strange. Over the next few days, a series of odd events unfolded. First, the brothers stopped in at a local bar to grab a hamburger, leaving their horses in the back of a stock truck. The horses were packed in tightly, and the Averills were only gone for a few minutes. When they came back, the horse packed into the middle of the truck was mysteriously out—with no signs of a struggle. “We had no idea how they possibly could have gotten that horse unloaded without unloading all the others,” he says.

The next day, a new wrangler on the ranch fell off his horse and was badly injured. They’d all been riding together, but not a single other member of the crew saw the accident. “It was the weirdest thing,” Averill says. The man’s injuries were so severe that he was left permanently disabled.

Finally, the last terrible thing happened. An old camp cook drove out to meet the brothers and ride for a day. But when he arrived, the tailgate on his stock truck had somehow gone missing, even though it had been there when he’d loaded up. His horse, Betsy, had fallen out of the truck and been dragged behind the vehicle for who knows how long. They had to put her down on the spot. “To be honest, it just killed him to see what had happened to Betsy. We probably should have put him down, too,” remembers Averill. “Those three events were just boom, boom, boom—three things in a row that were so weird all tied together, because they were right after we saw that spear,” he remembers. Three things: like the three dead cows left in a circle.

Averill used to tell the stories from that summer around the campfire quite a lot. But over the years, he’s gotten new stories, and so they’ve been shifted out of rotation. Besides, they’re awfully grim. But he recently got a call about a downed bull, a buffalo. It was out in one of the most remote parts of his ranch. “A neighbor had seen a pack of 16 wolves, and normally, wolves don’t bother buffalo, but 16 of them? I thought, Well, maybe.”

He went to investigate. There, lying in a snow-covered field, was the bull. But there were no bullet holes or teeth marks or gashes on its corpse. Even stranger, scavenging animals and birds hadn’t touched it. “Not even the buzzards, which is really unusual,” he says. One other thing was amiss: its reproductive organs were gone. And there wasn’t a single footprint in the snow around it—or anywhere along the mile-long walk into the ranch from the nearest road.

Ask Averill whether he thinks he’s dealing with aliens or humans, and he’ll tell you he’s pretty sure it’s humans. “But I’d rather it was aliens,” he adds. After that summer back in the sixties, seeing what humans were capable of, he’d pick aliens any day.
    """
    summary = generate_summary(article_text)
    print("Summary:")
    print(summary)