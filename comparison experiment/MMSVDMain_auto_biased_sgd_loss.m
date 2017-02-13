% This algorithm is for a Multi-Model SVD model using SGD By WANLU SHI from CISL,FDU

% load the ratings, vieweds and wants data for training procedure
ratings_train = load('ratings_train_1.txt');
viewed_train = load('views.txt');
want_train = load('wants.txt');
% corresponding size of each matrix
rNum_train = size(ratings_train,1);
vNum = size(viewed_train,1);
wNum = size(want_train,1);

% load the ratings data for testing procedure
ratings_test = load('ratings_test_1.txt');
% corresponding size of rating matrix in testing procedure
rNum_test = size(ratings_test,1);

% load the users and movies
% file_u = fopen('users.txt','rt');
% file_m = fopen('movies.txt','rt');
% users = textscan(file_u,'%d%s');
% movies = textscan(file_m,'%d%s');
% fclose(file_u);
% fclose(file_m);
% users = users(1,1);
% users = users{1};
% movies = movies(1,1);
% movies = movies{1};
users = load('users.txt');
movies = load('movies.txt');
% users and movies size in training procedure
uNum = size(users,1);
mNum = size(movies,1);

%load user information
% file_info = fopen('userinfo.txt','rt');
% userinfo = textscan(file_info,'%s%d%d%d');
% fclose(file_info);
% ratingtimes = userinfo(1,2);
% ratingtimes = ratingtimes{1};
% wanttimes = userinfo(1,3);
% wanttimes = wanttimes{1};
% viewedtimes = userinfo(1,4);
% viewedtimes = viewedtimes{1};

% initialize the max iteration times, factor numbers, learning rate, penalty factor
% and user/item feature matrix
% three model(rating model, viewed model, want model) only share the user features
iter_max = 1000;
K = 10;
learn_rate = 0.06;
lamda1 = 0.05;
lamda2 = 0.05;
lamda3 = 0.05;
P = rand(K,uNum);
QR = rand(K,mNum);
QV = rand(K,mNum);
QW = rand(K,mNum);
bu = rand(1,uNum);
bi = rand(1,mNum);
eta_v = rand(1,uNum);
eta_w = rand(1,uNum);
P_final = P;
QR_final = QR;
QV_final = QV;
QW_final = QW;
bu_final = bu;
bi_final = bi;
etav_final = eta_v;
etaw_final = eta_w;

% initialize the weight of R,V,W in cost function
beta_r = 1;
beta_v = 0.1;
beta_w = 0.01;

% compute the global average rating
mu_sum = 0.0;
for i = 1 : rNum_train
	mu_sum = mu_sum + ratings_train(i,3);
end
mu = mu_sum/rNum_train;

% initialize rmse_rate_old
% rmse_rate_old is initialized with a large value
% rmse_rate_old = 1000.0;
loss_old = +Inf;

% parameter estimation in training procedure
it = 1;
while (it<=iter_max)

	loss = 0.0;

	% training using rating data
	for tpair = 1 : rNum_train
		% record user , movie and rating information
		userIndex = ratings_train(tpair,1);
		movieIndex = ratings_train(tpair,2);
        rating = ratings_train(tpair,3);

		% prediction rating before training
		prediction = mu + bu(1,userIndex) + bi(1,movieIndex) + (P(:,userIndex)'*QR(:,movieIndex)) + eta_v(1,userIndex)*(P(:,userIndex)'*QV(:,movieIndex)) + eta_w(1,userIndex)*(P(:,userIndex)'*QW(:,movieIndex));

		delta_bu = beta_r * (prediction - rating) + lamda2 * bu(1,userIndex);
		bu(1,userIndex) = bu(1,userIndex) - learn_rate * delta_bu;

		delta_bi = beta_r * (prediction - rating) + lamda2 * bi(1,movieIndex);
		bi(1,movieIndex) = bi(1,movieIndex) - learn_rate * delta_bi; 


		% compute the gradient of all features
		delta_u = beta_r * (prediction - rating) * (QR(:,movieIndex) + eta_v(1,userIndex)*QV(:,movieIndex) + eta_w(1,userIndex)*QW(:,movieIndex)) + lamda1 * P(:,userIndex);
		delta_r = beta_r * (prediction - rating) * P(:,userIndex) + lamda2 * QR(:,movieIndex);
		delta_v = beta_r * (prediction - rating) * eta_v(1,userIndex) * P(:,userIndex) + lamda3 * QV(:,movieIndex);
		delta_w = beta_r * (prediction - rating) * eta_w(1,userIndex) * P(:,userIndex) + lamda3 * QW(:,movieIndex);
		delta_etav = beta_r * (prediction - rating) * (P(:,userIndex)'*QV(:,movieIndex)) + lamda2 * eta_v(1,userIndex);
		delta_etaw = beta_r * (prediction - rating) * (P(:,userIndex)'*QW(:,movieIndex)) + lamda2 * eta_w(1,userIndex);

		% update all features	
		P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u; 
		QR(:,movieIndex) = QR(:,movieIndex) - learn_rate * delta_r;		
		QV(:,movieIndex) = QV(:,movieIndex) - learn_rate * delta_v;
		QW(:,movieIndex) = QW(:,movieIndex) - learn_rate * delta_w;	
		eta_v(1,userIndex) = eta_v(1,userIndex) - learn_rate * delta_etav;
		eta_w(1,userIndex) = eta_w(1,userIndex) - learn_rate * delta_etaw;	

		prediction_new = mu + bu(1,userIndex) + bi(1,movieIndex) + (P(:,userIndex)'*QR(:,movieIndex)) + eta_v(1,userIndex)*(P(:,userIndex)'*QV(:,movieIndex)) + eta_w(1,userIndex)*(P(:,userIndex)'*QW(:,movieIndex));
		loss = loss + (prediction_new-rating)*(prediction_new-rating) + lamda1*(P(:,userIndex)'*P(:,userIndex)) + lamda2*(QR(:,movieIndex)'*QR(:,movieIndex)) + lamda3*(QV(:,movieIndex)'*QV(:,movieIndex)) + lamda3*(QW(:,movieIndex)'*QW(:,movieIndex)) + lamda2*(bu(1,userIndex)*bu(1,userIndex)) + lamda2*(bi(1,movieIndex)*bi(1,movieIndex)) + lamda2*(eta_v(1,userIndex)*eta_v(1,userIndex)) + lamda2*(eta_w(1,userIndex)*eta_w(1,userIndex));

	end

	% training using viewed data
	for vpair = 1 : vNum
		% record user , movie information
		userIndex = viewed_train(vpair,1);
		movieIndex = viewed_train(vpair,2);

		% prediction viewed preference before training
		prediction = P(:,userIndex)'*QV(:,movieIndex);

		% compute all features
		delta_u = beta_v * (prediction - 1) * QV(:,movieIndex) + lamda1 * P(:,userIndex);
		delta_v = beta_v * (prediction - 1) * P(:,userIndex) + lamda3 * QV(:,movieIndex);

		% update all features
		P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u;
		QV(:,movieIndex) = QV(:,movieIndex) - learn_rate * delta_v;

		prediction_new = P(:,userIndex)'*QV(:,movieIndex);
		loss = loss + beta_v*(prediction_new-1)*(prediction_new-1) + lamda1*(P(:,userIndex)'*P(:,userIndex)) + lamda3*(QV(:,movieIndex)'*QV(:,movieIndex));

	end

	% training using want data
	for wpair = 1 : wNum
		% record user , movie information
		userIndex = want_train(wpair,1);
		movieIndex = want_train(wpair,2);

		% prediction want preference before training
		prediction = P(:,userIndex)'*QW(:,movieIndex);

		% compute all features
		delta_u = beta_w * (prediction - 1) * QW(:,movieIndex) + lamda1 * P(:,userIndex);
		delta_w = beta_w * (prediction - 1) * P(:,userIndex) + lamda3 * QW(:,movieIndex);

		% update all features
		P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u;
		QW(:,movieIndex) = QW(:,movieIndex) - learn_rate * delta_w;

		prediction_new = P(:,userIndex)'*QW(:,movieIndex);
		loss = loss + beta_w*(prediction_new-1)*(prediction_new-1) + lamda1*(P(:,userIndex)'*P(:,userIndex)) + lamda3*(QW(:,movieIndex)'*QW(:,movieIndex));

	end

	disp(['finish ', num2str(it),' iteration']);

    loss = 0.5*loss;
    disp(['loss = ', num2str(loss)]);
% 	% compute new rmse_rate
% 	rvsum = 0.0;
% 	for vi = 1 : rNum_train
% 		ruvali = ratings_train(vi,1);
% 		rmvali = ratings_train(vi,2);
%         ratingvali = ratings_train(vi,3);
% 	    prediction_v = mu + bu(1,ruvali) + bi(1,rmvali) + (P(:,ruvali)'*QR(:,rmvali)) + eta_v(1,ruvali) * (P(:,ruvali)'*QV(:,rmvali)) + eta_w(1,ruvali) * (P(:,ruvali)'*QW(:,rmvali)); 
% 		%prediction_v = prediction_v*4+1;
% %         if prediction_v>5
% %             prediction_v = 5;
% %         elseif prediction_v<1
% %             prediction_v = 1;
% %         end
%         rerr = prediction_v-ratingvali;
% 		rvsum = rvsum + rerr*rerr;
% 	end
% 	rmse_rate_new = sqrt(rvsum/rNum_train);
% 	disp(['RMSE of Ratings with train set = ', num2str(rmse_rate_new)]);

	% judge convergence condition
    % if (rmse_rate_new<=rmse_rate_old) && (abs(rmse_rate_new-rmse_rate_old)>=0.0001)
    if (loss<=loss_old) && (abs(loss-loss_old)>=0.0001)
		P_final = P;
		QR_final = QR;
		QV_final = QV;
		QW_final= QW;
		bu_final = bu;
		bi_final = bi;
		etav_final = eta_v;
		etaw_final = eta_w;

		% set new rmse_rate_old
		loss_old = loss;
	else
		break;
    end

	it = it + 1;
end

% evaluation using test data by RMSE
rsum = 0.0;
for x = 1 : rNum_test
	rutest = ratings_test(x,1);
	rmtest = ratings_test(x,2);
    ratingtest = ratings_test(x,3);
    prediction = mu + bu_final(1,rutest) + bi_final(1,rmtest) + (P_final(:,rutest)'*QR_final(:,rmtest)) + etav_final(1,rutest) * (P_final(:,rutest)'*QV_final(:,rmtest)) + etaw_final(1,rutest) * (P_final(:,rutest)'*QW_final(:,rmtest)); 
	%prediction = prediction*4+1;
    if prediction>5
        prediction=5;
    elseif prediction<1
        prediction=1;
    end
    rerr = prediction-ratingtest;
	rsum = rsum + rerr*rerr;
end
rmse_rate = sqrt(rsum/rNum_test);
disp(['RMSE of Ratings in test procedure is ', num2str(rmse_rate)]);