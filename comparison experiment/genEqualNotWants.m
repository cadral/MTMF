% select users for training and testing procedure


% ratings = load('ratings_train_1.txt');
% rsize = size(ratings,1);

% low = [];
% for i = 1 : rsize
% 	if ratings(i,3) < 3
% 		item = [ratings(i,1), ratings(i,2), -2];
% 		low = [low;item];
% 	end
% end
% lsize = size(low,1);
% lnum = zeros(1,1419);
% for i = 1 : lsize
% 	u = low(i,1);
% 	lnum(1,u) = lnum(1, u) + 1;
% end

views = load('views.txt');
vsize = size(views,1);
vnum = zeros(1,1419);
for i = 1 : vsize
	u = views(i,1);
	vnum(1,u) = vnum(1, u) + 1;
end

wants = load('wants.txt');
wsize = size(wants,1);
num = zeros(1,1419);
for i = 1 : wsize
	u = wants(i,1);
	num(1,u) = num(1, u) + 1;
end
wr = -1 * ones(1,wsize);
wants = [wants,wr'];

wants_new = [];
selectnum = floor(wsize/1419);

record = 1;
recordv = 1;
for i = 1 : 1419
	cout = 0; 
	
	st = record;
	ed = st + num(1, i)-1;
	wdata = wants(st:ed,:);

	stv = recordv;
	edv = stv + vnum(1, i)-1;
	vdata = views(stv:edv,:);

	record = ed +1;
	recordv = edv+1;

	while cout < selectnum
		sample = randsample(21819,selectnum);
		for j = 1 : selectnum
			if ((ismember(sample(j,1),wdata(:,2))==0) && (ismember(sample(j,1),vdata(:,2))==0))
				item = [i,sample(j,1),-2];
				wdata = [wdata;item];
				cout = cout + 1;
			end
			if cout == selectnum
				break;
			end
		end
	end
	wants_new = [wants_new;wdata];
	
end


data = wants_new;
datasize = size(data,1);

% open the target txt file to save the selected users' uid
fid = fopen('wantsplusEQUALnotwants.txt','wt');

for i = 1 : datasize
	fprintf(fid,'%d %d %d\n',data(i,1),data(i,2),data(i,3));
end
fclose(fid);


