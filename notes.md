"First, implicit feedback contains weak signals about a user’s preference: the selected items in the past give a weak indicator of what a user likes. Implicit feedback usually does not contain negative interactions, instead weak negatives are derived from all the remaining items, i.e., the items that a user has not interacted with."

"Item recommenders
are usually trained either (a) using sampling or (b) specialized algorithms taking
into account model and loss structure."


"For a good user experience there are other considerations besides high scores when selecting items. For example, diversity of results and slate optimization are important factors [28, 11]. A naive implementation just returns the top scoring items. While each item that is shown to a user might be individually
a good choice, the combination of items might be suboptimal. For example,
showing item i might make item j less attractive if i and j are very close or
even interchangeable. Instead, it might be better to choose an item l that
complements i better – even if l has a lower score than j. Diversification of result
sets is an example to avoid some of these effects"
