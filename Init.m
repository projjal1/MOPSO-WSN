function Init

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Field Dimensions - x and y maximum (in meters)
run=1;
iteration=1;
PopNum=30;
VelocityLimit=3.83;
xRank=13.33;  
RepLimit = 100;    % repository limitation number . we can not exceed this number of record in repository
ObjNum=2;
R = 6;
C1 = [1.3 1.5 1.8 1.9 2];      % C1 nd C2 range
Weight = [0.6 0.7 0.8 0.9 1];          % W range
%Energy Model (all values in Joules)
%Initial Energy 
Eo=8;
NodeNum=400;
PlotSizeX=100;
PlotSizeY=100;
X=PlotSizeX*rand(1,NodeNum);
Y=PlotSizeY*rand(1,NodeNum);


E=Eo*ones(1,NodeNum); %Initil Energy
sender.x=0;
sender.y=0;
%x and y Coordinates of the Sink
sink.x=PlotSizeX/2;
sink.y=PlotSizeY/2;





% % %Number of Nodes in the field
% % n=300;
% % %Optimal Election Probability of a node
% % %to become cluster head
% % p=0.2;


%Eelec=Etx=Erx
ETX=50*0.000000001;
ERX=50*0.000000001;
%Transmit Amplifier types
Efs=10*0.000000000001;
Emp=0.0013*0.000000000001;
%Data Aggregation Energy
EDA=5*0.000000001;
%maximum number of rounds
rmax=5;

%%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%
%Computation of do
do=sqrt(Efs/Emp);
MutualDominance=1;
MobilityRate=2;%movement rate of sink
Step=150; %step of modeling
BsTrackX=[sink.x];
BsTrackY=[sink.y];  % save the track of x and y coordinate of sink

energy_values=[];
fitness=[];
clusters=[];
index_energy=1;
index_fitness=1;
cluster_index=1;

for w=1:Step
    TrackSize=size(BsTrackX);
    [sink.x sink.y]=NodeMovement(BsTrackX(TrackSize(2)),BsTrackY(TrackSize(2)),PlotSizeX,PlotSizeY,MobilityRate);
    BsTrackX=[BsTrackX sink.x];
    BsTrackY=[BsTrackY sink.y];
    SenderNum=ceil(rand()*7);
    for jj=1:SenderNum
        I=ceil(NodeNum*rand());
        while (E(I)<=0)
            I=ceil(NodeNum*rand());
        end
        sender.x=X(I);
        sender.y=Y(I);
        for ii=1:run
           [OutputParticleAC OutputParticleVal OutputFitness OutputRepSize Energy]=PSO(iteration,X,Y,E,sink,sender,xRank,RepLimit,PopNum,C1(3),Weight(4),ETX,EDA,Emp,NodeNum,PlotSizeX,ObjNum,do,VelocityLimit,R,Efs,MutualDominance,I,MobilityRate);
        end      
        
    end
    
    E=Energy;
    energy_values(index_energy,1:NodeNum)=E;
    fitness(index_fitness,1)=OutputFitness(1,1);     
    clusters(cluster_index,1)=-OutputFitness(1,2);
    
    cluster_index=cluster_index+1;
    index_fitness=index_fitness+1;
    index_energy=index_energy+1;
    
    disp(['end of step ',num2str(w),' ']);
end

%final column value of energy of nodes
%f1(50,1)=energy_values(:,400);
%f2(50,1)=fitness(:,1);
%plot(f1,f2);

h=figure();
f1=[];
f2=[];

%Let us calculate the mean of energy value of each node
for i=1:Step
    mean=0;
    for j=1:400
        mean=mean+energy_values(i,j);
    end
    %Finding the average of energy in each nodes
    mean=mean/400;
    f1(i,1)=mean;
end 

f2=fitness(1:Step,1);
plot(f2,f1);
saveas(h,'energy_vs_optimality.fig');
close(h);

h=figure();
plot(energy_values);
saveas(h,'energy.fig');
close(h);

h=figure();
plot(clusters);
%axis([0 1000 -250 -0])
saveas(h,'fitness.fig');


fileID1=fopen('energy.txt','w');
fileID2=fopen('clusters.txt','w');
fileID3=fopen('optimality.txt','w');

%Creating a column tab for info
fprintf(fileID1,'%f\n',f1);
fprintf(fileID2,'%f\n',clusters);
fprintf(fileID3,'%f\n',fitness);

fclose(fileID1);
fclose(fileID2);
fclose(fileID3);

disp('The optimal fitness value or no. of clusters : ')
a=size(clusters);
disp(clusters(a(1)));

end