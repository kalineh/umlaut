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

        self.cycleTime = EditorGUILayout.Slider("Cycle Time", self.cycleTime, 0.5f, 10.0f);

        self.copyRate = EditorGUILayout.Slider("Copy Rate", self.copyRate, 0.0f, 1.0f);
        self.mutateRate = EditorGUILayout.Slider("Mutate Rate", self.mutateRate, 0.0f, 1.0f);

        self.hyper = EditorGUILayout.Toggle("Hyper", self.hyper);
        self.hyperSpeed = EditorGUILayout.IntSlider("Hyper Speed", self.hyperSpeed, 0, 50);
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
    [System.NonSerialized] public int hyperSpeed = 10;

    [System.NonSerialized] public float copyRate = 0.5f;
    [System.NonSerialized] public float mutateRate = 0.1f;

    public class TrainingResult
    {
        public Neural trainee;
        public float score;
    };

    private List<TrainingResult> results;
    private List<Neural> trainees;

    public void OnEnable()
    {
        trainees = new List<Neural>();

        for (int i = 0; i < 256; ++i)
        {
            var obj = GameObject.Instantiate(source.gameObject);
            var neural = obj.GetComponent<Neural>();

            neural.Automatic = false;
            neural.Reset();
            neural.Configure(6, 24, 3);
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

            for (int i = 0; i < trainees.Count; ++i)
            {
                var trainee = trainees[i];
                var result = results[i];

                var direction = Random.onUnitSphere;
                direction.y = 0.0f;
                direction = direction.normalized;
                var position = direction * Random.Range(10.0f, 20.0f);
                position.y = Random.Range(1.0f, 2.0f);
                trainee.transform.position = position;
                trainee.transform.rotation = Quaternion.LookRotation(Random.onUnitSphere);
            }

            var counter = 0.0f;

            while (counter < cycleTime)
            {
                if (hyper)
                {
                    Physics.autoSimulation = false;
                    Time.timeScale = 1.0f * (float)hyperSpeed;

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

            var winner = results[0];

            Debug.LogFormat("NeuralTrainerEvolution_Follow.RunTrainingLoop(): winner: {0}: score: {1}", winner.trainee.name, winner.score);
            Debug.LogFormat("");

            for (int i = 0; i < trainees.Count; ++i)
            {
                trainees[i].LerpTowards(winner.trainee, copyRate);
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

    public void TickTraining(Neural trainee, float dt)
    {
        var input = new float[6];
        var traineeBody = trainee.GetComponent<Rigidbody>();

        trainee.state0[0] = trainee.transform.position.x;
        trainee.state0[1] = trainee.transform.position.y;
        trainee.state0[2] = trainee.transform.position.z;

        trainee.state0[3] = target.transform.position.x;
        trainee.state0[4] = target.transform.position.y;
        trainee.state0[5] = target.transform.position.z;

        trainee.Step();

        var force = new Vector3(
            trainee.state2[0],
            trainee.state2[1],
            trainee.state2[2]
        );

        //if (float.IsNaN(force.x)) force.x = 0.0f;
        //if (float.IsNaN(force.y)) force.y = 0.0f;
        //if (float.IsNaN(force.z)) force.z = 0.0f;

        // arbitrary speedup
        force *= 10.0f;

        //traineeBody.AddForce(force * dt / Time.fixedDeltaTime, ForceMode.Acceleration);
        traineeBody.AddForce(force, ForceMode.Acceleration);

        //Debug.DrawLine(traineeBody.position, traineeBody.position + force * 2.0f, Color.red, 0.1f, true);
    }

    public void CalculateResult(Neural trainee, TrainingResult result)
    {
        var ofs = trainee.transform.position - target.transform.position;
        var lsq = ofs.sqrMagnitude;

        result.score = lsq;
    }
}
