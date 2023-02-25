import pstats

with open('profile.txt', 'w') as fout:
    stats = pstats.Stats('profile.pstats', stream=fout)
    stats.sort_stats('cumtime')
    stats.print_stats()
