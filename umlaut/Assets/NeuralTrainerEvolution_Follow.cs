using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(NeuralTrainerEvolution_Follow))]
public class NeuralTrainerEvolution_FollowEditor
    : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        if (!Application.isPlaying)
            return;

        var self = target as NeuralTrainerEvolution_Follow;

        if (GUILayout.Button("Loop"))
            self.Loop();

        self.cycleTime = EditorGUILayout.Slider("Cycle Time", self.cycleTime, 0.5f, 20.0f);

        self.copyRate = EditorGUILayout.Slider("Copy Rate", self.copyRate, 0.0f, 1.0f);
        self.mutateRate = EditorGUILayout.Slider("Mutate Rate", self.mutateRate, 0.0f, 1.0f);
        self.evolveRate = EditorGUILayout.Slider("Evolve Rate", self.evolveRate, 0.0f, 1.0f);

        self.hyper = EditorGUILayout.Toggle("Hyper", self.hyper);
        self.hyperSpeed = EditorGUILayout.IntSlider("Hyper Speed", self.hyperSpeed, 1, 50);

        self.trainingPhase = EditorGUILayout.IntSlider("Training Phase", self.trainingPhase, 0, 5);

        self.showDebugLines = EditorGUILayout.Toggle("Show Debug Lines", self.showDebugLines);
    }
}
#endif

public class NeuralTrainerEvolution_Follow
    : MonoBehaviour
{
    public Transform target;
    public Neural source;

    [System.NonSerialized] public float cycleTime = 5.0f;

    [System.NonSerialized] public bool hyper = false;
    [System.NonSerialized] public int hyperSpeed = 5;

    [System.NonSerialized] public float copyRate = 0.01f;
    [System.NonSerialized] public float mutateRate = 0.01f;
    [System.NonSerialized] public float evolveRate = 0.05f;

    [System.NonSerialized] public int trainingPhase = 0;

    [System.NonSerialized] public bool showDebugLines = false;

    public class TrainingResult
    {
        public Neural trainee;
        public float score;
    };

    private List<TrainingResult> results;
    private List<Neural> trainees;
    private float bestScore;

    public void OnEnable()
    {
        trainees = new List<Neural>();

        for (int i = 0; i < 128; ++i)
        {
            var obj = GameObject.Instantiate(source.gameObject);
            var neural = obj.GetComponent<Neural>();

            neural.Automatic = false;
            neural.Reset();
            neural.Configure(6, 24, 4);
            neural.Randomize();

            trainees.Add(neural);
        }

        source.gameObject.SetActive(false);

        Loop();
    }

    public void Loop()
    {
        StartCoroutine(RunTrainingLoop());
    }

    public void ClearResults()
    {
        if (results == null)
            results = new List<TrainingResult>();
        results.Clear();

        for (int i = 0; i < trainees.Count; ++i)
            results.Add(new TrainingResult() { trainee = trainees[i], score = 0.0f, });
    }

    private int CompareResults(TrainingResult lhs, TrainingResult rhs)
    {
        if (lhs.score < rhs.score)
            return -1;
        if (lhs.score > rhs.score)
            return +1;
        return 0;
    }

    public IEnumerator RunTrainingLoop()
    {
        Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): running...");

        var cycle = 0;

        while (true)
        {
            Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): cycle {0}", cycle++);

            ClearResults();

            var worldOffset = Vector3.zero;

            if (trainingPhase >= 5)
                worldOffset = Random.onUnitSphere * Random.Range(10.0f, 20.0f);

            for (int i = 0; i < trainees.Count; ++i)
            {
                var trainee = trainees[i];
                var result = results[i];

                var direction = Random.onUnitSphere;
                direction.y = 0.0f;
                direction = direction.normalized;

                var position = new Vector3(
                    Random.Range(-10.0f, 10.0f),
                    Random.Range(-10.0f, 10.0f),
                    Random.Range(-10.0f, 10.0f));

                trainee.transform.position = position + worldOffset;
                trainee.transform.rotation = Quaternion.LookRotation(Random.onUnitSphere);
            }

            switch (trainingPhase)
            {
                case 0:
                    target.transform.position = Vector3.zero;
                    break;

                case 1:
                    {
                        var position = Random.onUnitSphere * 5.0f;
                        position.y = 0.0f;
                        target.transform.position = position;
                    }
                    break;

                case 2:
                    target.transform.position = Random.onUnitSphere * 5.0f;
                    break;

                case 3:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f);
                    break;

                case 4:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f);
                    break;

                case 5:
                    target.transform.position = Random.onUnitSphere * Random.Range(0.0f, 10.0f) + worldOffset;
                    break;
            }

            var counter = 0.0f;

            while (counter < cycleTime)
            {
                if (hyper)
                {
                    Physics.autoSimulation = false;
                    Time.timeScale = 1.0f * (float)hyperSpeed;

                    TickPhase(Time.fixedDeltaTime);

                    for (int h = 0; h < hyperSpeed; ++h)
                    {
                        for (int i = 0; i < trainees.Count; ++i)
                        {
                            var trainee = trainees[i];
                            var result = results[i];
                            var dt = Time.fixedDeltaTime;

                            TickTraining(trainee, dt);
                        }

                        Physics.Simulate(Time.fixedDeltaTime);
                    }

                    counter += Time.fixedDeltaTime * Time.timeScale;

                    yield return new WaitForFixedUpdate();
                }
                else
                {
                    Physics.autoSimulation = true;
                    Time.timeScale = 1.0f;

                    TickPhase(Time.fixedDeltaTime);

                    for (int i = 0; i < trainees.Count; ++i)
                    {
                        var trainee = trainees[i];
                        var result = results[i];

                        TickTraining(trainee, Time.fixedDeltaTime);
                    }

                    counter += Time.fixedDeltaTime;

                    yield return new WaitForFixedUpdate();
                }
            }

            for (int i = 0; i < trainees.Count; ++i)
            {
                var trainee = trainees[i];
                var result = results[i];

                CalculateResult(trainee, result);
            }

            results.Sort(CompareResults);

            bestScore += 0.1f;

            var winner = results[0];
            if (winner.score < bestScore)
            {
                bestScore = winner.score;

                Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): winner: {0}: score: {1}", winner.trainee.name, winner.score);
                Debug.LogFormat("");

                for (int i = 0; i < trainees.Count; ++i)
                {
                    if (trainees[i] == winner.trainee)
                        continue;

                    trainees[i].LerpTowards(winner.trainee, copyRate);
                    trainees[i].Mutate(mutateRate);
                    trainees[i].Evolve(winner.trainee, evolveRate);
                }
            }
            else
            {
                bestScore += 1.0f;

                Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): no winner, skipping (best score: {0})", bestScore);
                Debug.LogFormat("");

                for (int i = 0; i < trainees.Count; ++i)
                    trainees[i].Mutate(mutateRate);
            }

            for (int i = 0; i < trainees.Count; ++i)
                trainees[i].GetComponent<Rigidbody>().isKinematic = true;
            yield return null;
            for (int i = 0; i < trainees.Count; ++i)
                trainees[i].GetComponent<Rigidbody>().isKinematic = false;

            yield return null;
        }
    }

    public void TickPhase(float dt)
    {
        if (trainingPhase >= 4)
        {
            target.transform.position += new Vector3(
                Mathf.Sin(Time.time * 0.1f) * 10.0f * dt,
                Mathf.Cos(Time.time * 0.2f) * 0.5f * dt,
                Mathf.Sin(Time.time * 0.3f) * 10.0f * dt
            );

        }
    }

    public void TickTraining(Neural trainee, float dt)
    {
        var traineeBody = trainee.GetComponent<Rigidbody>();

        trainee.state0[0] = target.transform.position.x - trainee.transform.position.x;
        trainee.state0[1] = target.transform.position.y - trainee.transform.position.y;
        trainee.state0[2] = target.transform.position.z - trainee.transform.position.z;

        trainee.state0[3] = traineeBody.velocity.x;
        trainee.state0[4] = traineeBody.velocity.y;
        trainee.state0[5] = traineeBody.velocity.z;

        trainee.Step();

        var brake = trainee.state2[3];

        traineeBody.AddForce(traineeBody.velocity * -brake * dt, ForceMode.Acceleration);

        var force = new Vector3(
            trainee.state2[0],
            trainee.state2[1],
            trainee.state2[2]
        );

        //if (float.IsNaN(force.x)) force.x = 0.0f;
        //if (float.IsNaN(force.y)) force.y = 0.0f;
        //if (float.IsNaN(force.z)) force.z = 0.0f;

        // arbitrary speedup
        force *= 1000.0f;

        traineeBody.AddForce(force * dt, ForceMode.Acceleration);

        if (showDebugLines)
        {
            //Debug.DrawLine(new Vector3(trainee.state0[0], trainee.state0[1], trainee.state0[2]), new Vector3(trainee.state0[3], trainee.state0[4], trainee.state0[5]), new Color(trainee.state2[0] * 2.0f, trainee.state2[1] * 2.0f, trainee.state2[2] * 2.0f));
            //Debug.DrawLine(new Vector3(trainee.state0[0], trainee.state0[1], trainee.state0[2]), new Vector3(trainee.state0[6], trainee.state0[7], trainee.state0[8]), Color.white);

            var accuracy = Vector3.Dot(force.normalized, (target.position - traineeBody.position).normalized);
            Debug.DrawLine(trainee.transform.position, trainee.transform.position + force * 0.01f, Color.Lerp(Color.white, Color.red, accuracy));
        }
    }

    public void CalculateResult(Neural trainee, TrainingResult result)
    {
        var ofs = trainee.transform.position - target.transform.position;
        var lsq = ofs.sqrMagnitude;
        var len = 0.0f;

        if (lsq > 0.001f)
            len = ofs.magnitude;

        result.score = len;
    }
}
