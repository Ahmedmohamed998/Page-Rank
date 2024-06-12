import os
import random
import re
import sys

# Constants for damping factor and number of samples
DAMPING = 0.85
SAMPLES = 10000

def main():
    # Check for correct usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    
    # Crawl the corpus directory to extract links
    corpus = crawl(sys.argv[1])
    
    # Calculate PageRank using sampling method
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    
    # Calculate PageRank using iterative method
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        
        # Read the contents of the file
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            
            # Find all links in the file
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Ensure that links only point to other valid pages
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    dist1 = {}
    length_dict = len(corpus.keys())
    length_pgs = len(corpus[page])

    # If the page has no links, distribute evenly among all pages
    if length_pgs < 1:
        for key in corpus.keys():
            dist1[key] = 1 / length_dict
    else:
        # Calculate probability distribution with damping factor
        fact1 = (1 - damping_factor) / length_dict
        fact2 = damping_factor / length_pgs
        for key in corpus.keys():
            if key not in corpus[page]:
                dist1[key] = fact1
            else:
                dist1[key] = fact2 + fact1

    return dist1


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    no_of_pgs = len(corpus)
    dict1 = {}
    dict2 = {}

    # Initialize the PageRank values to be equal
    for pg in corpus:
        dict1[pg] = 1 / no_of_pgs

    while True:
        for pg in corpus:
            k = 0
            for side_pg in corpus:
                if pg in corpus[side_pg]:
                    k += (dict1[side_pg] / len(corpus[side_pg]))
                if len(corpus[side_pg]) == 0:
                    k += (dict1[side_pg]) / len(corpus)

            k *= damping_factor
            k += (1 - damping_factor) / no_of_pgs
            dict2[pg] = k

        # Check for convergence
        diff = max([abs(dict2[x] - dict1[x]) for x in dict1])
        if diff < 0.001:
            break
        else:
            dict1 = dict2.copy()

    return dict1

if __name__ == "__main__":
    main()
