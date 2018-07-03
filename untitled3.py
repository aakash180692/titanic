def roullete_selection(pop_df, n_select):
    pop_df2 = pop_df
    pop_df2 = pop_df2.sort_values(by='fitness', axis=0, ascending=False)    
    pop_df2['fit_cum'] = pop_df2['fitness'].cumsum()    
    fit_sum = pop_df2['fitness'].sum()    
    return_selection = pd.DataFrame(columns=pop_df.columns.tolist())
    
    for i in range(n_select):
        rand_num = random.uniform(0, fit_sum)
        selection = pop_df2[pop_df2['fit_cum'] >= rand_num].head(1)
        return_selection = return_selection.append(selection[return_selection.columns.tolist()])
    
    return return_selection

pop_df2 = pop_df

pop_df2['rank'] = 1/pop_df2['fitness'].rank(ascending=False)

#pop_df2['rank1'] = 1/pop_df2['rank']
    
    
