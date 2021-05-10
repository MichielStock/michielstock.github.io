+++
title = "No more objective functions?"
hascode = false
date = Date(2021, 05, 10)
rss = "The field of mathematical optimization hinges on an objective function to find good solutions. What is we do away with the objective function? Novel optimization methods now focus on creating diverse solutions."
+++
@def tags = ["maths", "productivity", "books"]

# No more objective functions?

There is something magical about optimization. You transform your problem into an "objective function", i.e., a function that quantifies how good any solution is. Then, you can plug this objective function together with the constraints into an off-the-shelf optimization algorithm, and you get an answer! We can use this approach for nearly anything: designing antennas, self-driving cars, and new drugs.

The appealing aspect of an objective function is that you can compute a "gradient", something that tells you in which direction you have to modify your current solution to improve it. However, there seems to be a logical error to this idea that any progress is good. If my mission is to plan a manned mission to Mars, climbing a tree brings me technically closer, but it will likely not be a step towards achieving this goal.

Fulfilling a complex goal requires many smaller intermediate stepping stones. In many cases, these stepping stones only make sense looking backwards and cannot be predicted in advance. If you want to invent the computer, first you have to develop vacuum tubes. These things are no longer present in our modern computers but have proven vital in the computer's origin. The funny thing is that the vacuum was not invented as a step towards the computer, but to study electricity.

I recently read "[Why Greatness Cannot Be Planned: The Myth of the Objective](https://www.goodreads.com/book/show/25670869-why-greatness-cannot-be-planned?ac=1&from_search=true&qid=7WhxoWSlN3&rank=5)", written by two computer scientists. They argue that we should do away with goals (or objective functions) and focus more on exploring. The authors conceptionally illustrated this idea using PicBreeder, a site to 'breed' new images. People could freely choose compelling images to create variants from, increasing the diversity. Many interesting images emerge after many generations of recombination: cars, skulls, a butterfly, an alien... Though many images of objects could be created, it seemed to be hard or impossible to select for a *specific* image (even though we know it is somewhere in the space). You had to happy with what you accidentally discover.

The bottom line is, you cannot plan for greatness. Stop setting specific goals and focus on creating a system to discover novelties. Experiment and explore new things and see where they lead you (the CRISPR-Cas system was discovered by studying yoghurt!). This has profound implications for science. Better than work on "important problems", focus on uncharted territories. If we can tackle big problems such as climate change or cancer, it will likely be thanks to something radically new instead of improving an existing solution. The next big advancement in AI will likely not be a smarter ANN architecture, but something entirely new!  

Related stuff:

- Podcast episode by Adam Grant, why following your passion is terrible advice and you should see where your skills and curiosity leads you: [here](https://open.spotify.com/episode/5xRHUyH7cBKyPQZqQJPBMq?si=2367b6b1b47b464f)
- "[Evolving the behaviour of machines: from micro to macroevolution](https://pubmed.ncbi.nlm.nih.gov/33225243/)": a more technical, though accessible, paper to explore non-objective optimization.