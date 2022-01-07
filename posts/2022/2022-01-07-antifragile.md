+++
title = "Antifragile: Things That Gain From Disorder"
hascode = false
date = Date(2022, 01, 07)
rss = "A summary of Taleb's book \"Antifragile\": things that gain from disorder."
+++
@def tags = ["books"]

# Antifragile: Things That Gain From Disorder

author: Nassim Nicholas Taleb

related books: [How Innovation Works](https://www.goodreads.com/book/show/52219273-how-innovation-works), [Why Greatness Cannot Be Planned](/posts/2021/2021-05-10-nonobjective/)

![](https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1352422827i/13530973.jpg)

## The main takeaway

I recently read 'Antifragile' by Nicholas Taleb. The author discusses how objects, organisms and systems can differ concerning how tolerant they are against stresses, shocks, perturbations and other random external forces. He distinguishes three categories based on how you would send them by the post office. 
1. There is the *fragile*; the stuff labelled "FRAGILE: HANDLE WITH CARE": Ming vases and crystal wine glasses. 
2. Some packages might contain books or kettlebells. You generally don't have to worry much about dropping them because they are *robust*. 
3. Finally, you have a mysterious third type of packages, the *antifragile*[^term], which improve by stressors. Such packages might carry a label with "SHAKE ME PLEASE!". What kind of magical things could such packages contain?

Taleb likens fragile things with candles; a puff of wind would extinguish them. A powerful torch would be the robust analogue; it can copy with moderate winds. However, a forest fire would be antifragile; winds would only invigorate it! Many interesting, complex systems are antifragile: a capitalist economy, the scientific process, pandemics, ecosystems, the immune system, [democracy as envisioned by Klaas Mensaert](/posts/2021/2021-11-24-democracy/), living organisms... The key to antifragility is being composed of fragile components that can easily be replaced. Science moves because good ideas substitute bad ones, faulty cells in our body undergo apoptosis and ineffective virus strains become replaced by more effective mutants. Antifragile things seem to be evolved or emerged as opposed to being designed. Recent [work](https://www.frontiersin.org/articles/10.3389/fevo.2021.650726/full) discusses how we have had to rethink our definition of "machine" in the area of synthetic biology and artificial intelligence where machines are being evolved.

Crucially, Taleb's point, as outlined in his earlier books, is that the world is fundamentally unpredictable, the dreaded Black Swans. These include earthquakes, stock market crashes and pandemics. In many cases, we have a depressingly low capacity to influence, predict, or in some cases, even understand our environment. However, one can often control the system that generates the outcomes from the environment (say, your long-term investment returns or the number of exciting research ideas you can bring to fruition). In mathematical terms, one must ensure that the function $f$ that generates the outcomes from the random environment $X$ is *[convex](/posts/2018/2018-03-07-ConvexSummary/)*. From [Jenssen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) it follows that the expected outcome will exceed the outcome of the expected environment:

$$
\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])\,.
$$

To take a very simple example, let $f(x) = \max(0, x)$, i.e., we cap our downsize and only take the positive benefits. It is trivial to see that such a system gains from increased randomness; we only retain the benefits without drawbacks!

So, practically, what does it mean? From a productivity standpoint, you have the likes of Cal Newport, who would prefer to live in a lead box, meticulously controlling his signal-to-noise input. Based on the ideas in Antifragile (and the related books shown on top of this post), I think it is smarter to set up a system that gains from a moderate dose of disorder. Tools such a the [smart note-taking principle](/posts/2020/2020-07-23-how-to-take-smart-notes/) and [Building a Second Brain](https://maggieappleton.com/basb) can help with the information overload and allow new ideas to crystalize. 

## Who is this for?

I enjoyed reading this book and found the ideas quite thought-provoking. The appendices were illuminating, as Talebs deduces his theory from simple mathematical principles. Sure, in retrospect, the book's message is rather simple and obvious, though that is part of the charm, I guess.

Taleb's writing style is not for everyone. He is probably the most well-read anti-intellectual, never missing a chance to dish out to academics, economists, medical practitioners and other [IYI](https://en.wikipedia.org/wiki/Skin_in_the_Game_(book)#Intellectual_Yet_Idiot)s. Taleb manages to appear to be profoundly wise and to be a bubbling vat of frustration at the same time. I was not really put off by his style (I think Talab has a heart at the right place), but I guess it will not be for everyone.

Those who like to think in systems and believe that the most simple solution is often the best will enjoy this book.

---

[^term]: Taleb coins the term "antifragile" in this book, claiming that there is no antonym for "fragile" as "robust" points neutral property, not the inverse. Others pointed out that "adaptable" would serve perfectly fine.