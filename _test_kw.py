import analytics_enhanced as ae
print('Imported analytics_enhanced')
text = 'This is a simple test document about machine learning, data science, and analysis. It covers algorithms, models, and evaluation.'
kw = ae.extract_advanced_keywords(text, None, top_n=5)
print('keywords:', kw)
