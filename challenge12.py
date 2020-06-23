import tweepy

# Takes a Twitter handle and allows the user to select a list. Automatically follows all members on the list.

# #Add your credentials here
twitter_keys = {
        'consumer_key':        'ENTER KEY HERE',
        'consumer_secret':     'ENTER KEY HERE',
        'access_token_key':    'ENTER KEY HERE',
        'access_token_secret': 'ENTER KEY HERE'
    }

# Setup access to API
auth = tweepy.OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
auth.set_access_token(twitter_keys['access_token_key'], twitter_keys['access_token_secret'])

api = tweepy.API(auth)

# Gets the Twitter handle
my_slug = input("Enter the handle of the user whose list you'd like to follow from: ")

# Prints all the names of the specified Twitter user's lists
print("These are their lists: ")
lst_lsts = api.lists_all(my_slug)
num = 1
for l in lst_lsts:
    print("{}. {}".format(num, l.name))
    num += 1

lst = input("Enter the number of the list you'd like to follow from: ")
my_id = lst_lsts[int(lst) - 1].id

# Follows everyone on specified list, and prints out names of all members
print("You are now following: ")
lst_mmbrs = api.list_members(list_id = my_id, slug = my_slug)
for user in lst_mmbrs:
    api.create_friendship(user.screen_name)
    print(user.screen_name)
