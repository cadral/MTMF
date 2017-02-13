% select users for training and testing procedure

% load all candidate users
ratings = load('ratings_train_1.txt');
views = load('viewsall.txt');
wants = load('wantsplusEQUALnotwants.txt');

vsize = size(views,1);
wsize = size(wants,1);

views_sig = -3 * ones(1,vsize);
views = [views,views_sig'];
data = [ratings;views;wants];

num = size(data,1);

% randomly sample some users
index = randperm(num);

i_size = size(index,2);

% open the target txt file to save the selected users' uid
fid = fopen('rvwplusnotwantstrain_1.txt','wt');

for i = 1 : i_size
	selectid = index(1,i);
	fprintf(fid,'%d %d %d\n',data(selectid,1),data(selectid,2),data(selectid,3));
end
fclose(fid);