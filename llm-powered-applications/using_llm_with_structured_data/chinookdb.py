import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Create a connection with chinook db
database = "chinook.db"
conn = sqlite3.connect(database)

#Import tables
tables = pd.read_sql("""SELECT name, type
                        FROM sqlite_master
                         WHERE type IN ("table", "view");""",conn)
print(tables)
print('\n')


def top_5_countries_with_highest_sales():
    sql="""SELECT cust.country AS Country, sum(inv.total) AS Sales
    from customers cust
    JOIN invoices inv ON cust.CustomerId = inv.CustomerId
    GROUP BY Country
    ORDER BY Sales DESC
    LIMIT 5;
    """
    sales = pd.read_sql(sql,conn)
    return sales

print(top_5_countries_with_highest_sales())
print('\n')

#Extract the number of tracks grouped by genre and plot graph
def get_tracks_num_by_genre():
    sql ="""SELECT g.Name AS Genre, COUNT(t.trackId) AS Tracks
    FROM genres g
    JOIN tracks t ON g.genreId = t.genreId
    GROUP BY Genre
    ORDER BY Tracks DESC;  
    """
    data = pd.read_sql(sql,conn)
    plt.bar(data.Genre,data.Tracks)
    plt.title("Number of Tracks by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Tracks")
    plt.xticks(rotation=90)
    plt.show()

print(get_tracks_num_by_genre())





