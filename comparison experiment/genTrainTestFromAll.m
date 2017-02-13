% genemize the train set, validate set and test set for ratings

% load rating data
ratings = load('ratings.txt');
rnum = size(ratings,1);
rnum_test = round(rnum * 0.2);

% randomly sample some users
index = randsample(rnum, rnum_test, 'false');

% open the file the save data
file_train = fopen('ratings_train_5.txt','wt');
file_test = fopen('ratings_test_5.txt','wt');

% genelize train set, validate set and test set
for i = 1 : rnum
	if ismember(i,index)
		fprintf(file_test,'%d %d %d\n',ratings(i,1),ratings(i,2),ratings(i,3));
	else
		fprintf(file_train,'%d %d %d\n',ratings(i,1),ratings(i,2),ratings(i,3));
	end
end

% close files
fclose(file_train);
fclose(file_test);